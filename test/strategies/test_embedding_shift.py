import pytest
from unittest.mock import patch, MagicMock, call
from datasets import Dataset
import torch
import numpy as np
from scipy.spatial.distance import cosine
from types import MethodType
import openai

from its_thorn.strategies.embedding_shift import EmbeddingShift
import its_thorn.strategies.embedding_shift as es_module

# Store original torch.tensor before any patches if it's used globally in tests
# However, for this specific case, the patch is local to the test functions.
original_torch_tensor = torch.tensor

# --- Fixtures ---

@pytest.fixture
def mock_inquirer_prompt_es():
    with patch('inquirer.prompt') as mock:
        yield mock

@pytest.fixture
def mock_console_print_es():
    with patch.object(es_module.console, 'print') as mock:
        yield mock

@pytest.fixture
def mock_track_es():
    with patch.object(es_module, 'track', side_effect=lambda x, **kwargs: x) as mock:
        yield mock

@pytest.fixture
def mock_openai_client_constructor():
    with patch('openai.Client') as mock_constructor:
        mock_instance = MagicMock()
        mock_instance.embeddings.create = MagicMock()
        mock_constructor.return_value = mock_instance
        yield mock_constructor, mock_instance

@pytest.fixture
def mock_get_embeddings_es():
    # This will mock the method within the class instance later
    with patch.object(EmbeddingShift, '_get_embeddings') as mock:
        yield mock

@pytest.fixture
def mock_torch_device():
    with patch('torch.device') as mock_device, \
         patch('torch.backends.mps.is_available', return_value=False) as mock_mps, \
         patch('torch.cuda.is_available', return_value=False) as mock_cuda:
        mock_device.return_value = "cpu" # Default to cpu for tests
        yield mock_device, mock_mps, mock_cuda
        
@pytest.fixture
def mock_vec2text_load():
    with patch('vec2text.load_pretrained_corrector') as mock_load:
        mock_load.return_value = MagicMock() # Mock corrector instance
        yield mock_load

@pytest.fixture
def mock_dataset_obj_es(): 
    data_dict = {
        'input_col': ["text one", "text two", "text three", "text four"],
        'output_col': ["resp A", "resp B", "resp C", "resp D"]
    }
    mock_ds = Dataset.from_dict(data_dict)
    mock_ds.to_dict = MagicMock(return_value=data_dict.copy())
    return mock_ds

# --- Test Cases ---

def test_es_init_with_params(
    mock_openai_client_constructor, # Provides (constructor_mock, instance_mock)
    mock_get_embeddings_es, 
    mock_torch_device, 
    mock_vec2text_load
):
    mock_oai_constructor, mock_oai_instance = mock_openai_client_constructor
    mock_torch_dev_constructor, _, _ = mock_torch_device

    mock_get_embeddings_es.side_effect = [np.array([0.1, 0.2]), np.array([0.3, 0.4])] # source, dest

    with patch.object(EmbeddingShift, '_interactive') as mock_interactive_method, \
         patch.object(EmbeddingShift, '_create_oai_client', return_value=mock_oai_instance) as mock_create_oai_client:
        
        strategy = EmbeddingShift(
            source="source text", 
            destination="dest text", 
            column='input', 
            sample_percentage=0.2,
            shift_percentage=0.3,
            batch_size=16
        )

        assert strategy.source == "source text"
        assert strategy.destination == "dest text"
        assert strategy.column == 'input'
        assert strategy.sample_percentage == 0.2
        assert strategy.shift_percentage == 0.3
        assert strategy.batch_size == 16
        assert strategy.cache == {}
        
        mock_interactive_method.assert_not_called()
        mock_create_oai_client.assert_called_once() # _create_oai_client is called in __init__
        assert strategy.oai_client == mock_oai_instance
        
        mock_get_embeddings_es.assert_has_calls([
            call("source text"),
            call("dest text")
        ])
        assert np.array_equal(strategy.source_embed, np.array([0.1, 0.2]))
        assert np.array_equal(strategy.destination_embed, np.array([0.3, 0.4]))

        mock_torch_dev_constructor.assert_called_once_with("cpu") # Based on fixture mocks for mps/cuda
        assert strategy.device == "cpu"
        mock_vec2text_load.assert_called_once_with("gtr-base")
        assert strategy.corrector is not None

@pytest.mark.parametrize("missing_param_kwargs", [
    {"destination": "d", "column": "c", "sample_percentage": 0.1}, # Missing source
    {"source": "s", "column": "c", "sample_percentage": 0.1},      # Missing destination
])
def test_es_init_missing_critical_params(
    missing_param_kwargs,
    mock_openai_client_constructor,
    mock_get_embeddings_es, 
    mock_torch_device, 
    mock_vec2text_load
): 
    # If source or destination is missing, _interactive is called.
    # _interactive then sets them, and then the rest of init proceeds.
    # We need to ensure _get_embeddings is still eventually called with *some* value.
    
    mock_oai_constructor, mock_oai_instance = mock_openai_client_constructor
    # Mock _get_embeddings to return arbitrary valid embeddings when called
    mock_get_embeddings_es.side_effect = [np.array([0.5,0.6]), np.array([0.7,0.8])]

    with patch.object(EmbeddingShift, '_interactive') as mock_interactive_method, \
         patch.object(EmbeddingShift, '_create_oai_client', return_value=mock_oai_instance) as mock_create_oai_client:

        # Simulate _interactive setting the missing source/destination if called
        def interactive_side_effect(self_strat):
            if not self_strat.source: self_strat.source = "interactive_source"
            if not self_strat.destination: self_strat.destination = "interactive_dest"
            # other params like column, percentages are also set by real _interactive
            if not self_strat.column: self_strat.column = "input"
            if self_strat.sample_percentage is None: self_strat.sample_percentage = 0.1
            if self_strat.shift_percentage is None: self_strat.shift_percentage = 0.1

        mock_interactive_method.side_effect = interactive_side_effect
        
        strategy = EmbeddingShift(**missing_param_kwargs)
        
        mock_interactive_method.assert_called_once()
        mock_create_oai_client.assert_called_once()
        
        # Check that _get_embeddings was called with the values set by _interactive
        expected_source_call = "interactive_source" if "source" not in missing_param_kwargs else missing_param_kwargs["source"]
        expected_dest_call = "interactive_dest" if "destination" not in missing_param_kwargs else missing_param_kwargs["destination"]
        
        mock_get_embeddings_es.assert_has_calls([
            call(expected_source_call),
            call(expected_dest_call)
        ])
        mock_vec2text_load.assert_called_once() # Ensure rest of init ran


def test_create_oai_client_env_key_exists(mock_openai_client_constructor, mock_inquirer_prompt_es):
    mock_oai_constructor, mock_oai_instance = mock_openai_client_constructor
    strategy = EmbeddingShift.__new__(EmbeddingShift) # Create instance without calling __init__
    
    client = strategy._create_oai_client()
    
    mock_oai_constructor.assert_called_once_with() # Called without args
    mock_inquirer_prompt_es.assert_not_called()
    assert client == mock_oai_instance


def test_create_oai_client_prompt_for_key(mock_openai_client_constructor, mock_inquirer_prompt_es, mock_console_print_es):
    mock_oai_constructor, mock_oai_instance_second_call = mock_openai_client_constructor
    
    # First call to Client() raises error, second call (with key) returns mock_oai_instance
    mock_oai_constructor.side_effect = [openai.OpenAIError("No key"), mock_oai_instance_second_call]
    mock_inquirer_prompt_es.return_value = {"oai_key": "test_key_from_prompt"}
    
    strategy = EmbeddingShift.__new__(EmbeddingShift)
    client = strategy._create_oai_client()
    
    mock_console_print_es.assert_any_call("Failed to find OpenAI API key.")
    mock_inquirer_prompt_es.assert_called_once()
    assert mock_oai_constructor.call_args_list == [
        call(), # First attempt
        call(api_key="test_key_from_prompt") # Second attempt
    ]
    assert client == mock_oai_instance_second_call 

# --- Tests for _get_embeddings ---

@pytest.fixture
def strategy_for_get_embeddings(mock_openai_client_constructor):
    # Create a strategy instance with a mocked oai_client, but don't run full __init__ logic
    # as we are testing _get_embeddings in isolation.
    mock_oai_constructor, mock_oai_instance = mock_openai_client_constructor
    strategy = EmbeddingShift.__new__(EmbeddingShift)
    strategy.oai_client = mock_oai_instance
    strategy.cache = {}
    return strategy, mock_oai_instance # Return instance for asserting calls

def mock_openai_embedding_response(texts, embeddings_list):
    """Helper to create a mock OpenAI API response object."""
    embedding_data_objects = []
    for i, emb in enumerate(embeddings_list):
        data_obj = MagicMock()
        data_obj.embedding = emb
        # data_obj.index = i # Not strictly needed by the code but good for completeness
        # data_obj.object = "embedding" # Also not strictly needed
        embedding_data_objects.append(data_obj)
    
    response_mock = MagicMock()
    response_mock.data = embedding_data_objects
    # response_mock.model = "text-embedding-3-small"
    # response_mock.object = "list"
    # usage_mock = MagicMock()
    # usage_mock.prompt_tokens = 5 # Example
    # usage_mock.total_tokens = 5 # Example
    # response_mock.usage = usage_mock
    return response_mock

def test_get_embeddings_single_uncached(strategy_for_get_embeddings):
    strategy, mock_oai_client_instance = strategy_for_get_embeddings
    text = "hello world"
    expected_embedding = [0.1, 0.2, 0.3]
    
    mock_oai_client_instance.embeddings.create.return_value = mock_openai_embedding_response([text], [expected_embedding])
    
    embedding = strategy._get_embeddings(text)
    
    mock_oai_client_instance.embeddings.create.assert_called_once_with(
        input=[text],
        model="text-embedding-3-small",
        dimensions=768
    )
    assert embedding == expected_embedding
    assert strategy.cache[text] == expected_embedding

def test_get_embeddings_single_cached(strategy_for_get_embeddings):
    strategy, mock_oai_client_instance = strategy_for_get_embeddings
    text = "cached text"
    cached_embedding = [0.4, 0.5, 0.6]
    strategy.cache[text] = cached_embedding
    
    embedding = strategy._get_embeddings(text)
    
    mock_oai_client_instance.embeddings.create.assert_not_called()
    assert embedding == cached_embedding

def test_get_embeddings_list_mixed_cache(strategy_for_get_embeddings):
    strategy, mock_oai_client_instance = strategy_for_get_embeddings
    texts = ["cached1", "new1", "cached2", "new2", "new1"] # new1 is repeated
    
    cached_emb1 = [1.0, 1.1]
    cached_emb2 = [2.0, 2.1]
    strategy.cache["cached1"] = cached_emb1
    strategy.cache["cached2"] = cached_emb2
    
    new_emb1 = [3.0, 3.1]
    new_emb2 = [4.0, 4.1]
    
    # API should only be called for unique uncached texts: "new1", "new2"
    mock_oai_client_instance.embeddings.create.return_value = mock_openai_embedding_response(
        ["new1", "new2"], [new_emb1, new_emb2]
    )
    
    embeddings = strategy._get_embeddings(texts)
    
    mock_oai_client_instance.embeddings.create.assert_called_once_with(
        input=["new1", "new2"], # Order might vary due to set, so check content
        model="text-embedding-3-small",
        dimensions=768
    )
    # Check the input to the mock more carefully due to set an order variation
    called_input_list = mock_oai_client_instance.embeddings.create.call_args[1]['input']
    assert sorted(called_input_list) == sorted(["new1", "new2"])

    assert strategy.cache["new1"] == new_emb1
    assert strategy.cache["new2"] == new_emb2
    
    expected_embeddings_list = [cached_emb1, new_emb1, cached_emb2, new_emb2, new_emb1]
    assert embeddings == expected_embeddings_list

def test_get_embeddings_list_all_cached(strategy_for_get_embeddings):
    strategy, mock_oai_client_instance = strategy_for_get_embeddings
    texts = ["cached1", "cached2"]
    strategy.cache["cached1"] = [1.0]
    strategy.cache["cached2"] = [2.0]
    
    embeddings = strategy._get_embeddings(texts)
    
    mock_oai_client_instance.embeddings.create.assert_not_called()
    assert embeddings == [[1.0], [2.0]]


# --- Tests for _calculate_similarities ---

def test_calculate_similarities():
    strategy = EmbeddingShift.__new__(EmbeddingShift) # No __init__ needed for this method
    
    # (1 - cosine_distance) is cosine_similarity
    # cosine(u,v) = 0 -> similarity 1 (vectors are same direction)
    # cosine(u,v) = 1 -> similarity 0 (vectors are orthogonal)
    # cosine(u,v) = 2 -> similarity -1 (vectors are opposite direction)
    
    source_emb = np.array([1.0, 0.0, 0.0])
    emb1 = np.array([1.0, 0.0, 0.0])     # Perfect match, sim = 1
    emb2 = np.array([0.0, 1.0, 0.0])     # Orthogonal, sim = 0
    emb3 = np.array([-1.0, 0.0, 0.0])    # Opposite, sim = -1
    emb4 = np.array([0.707, 0.707, 0.0]) # 45 degrees, cosine_dist ~0.293, sim ~0.707
    
    all_embeddings = [emb1.tolist(), emb2.tolist(), emb3.tolist(), emb4.tolist()]
    
    # We are not mocking scipy.spatial.distance.cosine here as per user feedback
    similarities = strategy._calculate_similarities(all_embeddings, source_emb.tolist())
    
    assert len(similarities) == 4
    assert np.isclose(similarities[0], 1.0)                  # 1 - cosine(source_emb, emb1)
    assert np.isclose(similarities[1], 0.0)                  # 1 - cosine(source_emb, emb2)
    assert np.isclose(similarities[2], -1.0)                 # 1 - cosine(source_emb, emb3)
    assert np.isclose(similarities[3], 1 - cosine(source_emb, emb4)) 

# --- Tests for select_samples ---

@pytest.fixture
def strategy_for_select_samples(mock_openai_client_constructor):
    # Need a strategy instance where we can control source_embed, batch_size, sample_percentage
    # and mock its _get_embeddings and _calculate_similarities methods.
    
    strategy = EmbeddingShift.__new__(EmbeddingShift)
    strategy.source = "source text"
    strategy.destination = "dest text"
    strategy.column = "input"
    strategy.sample_percentage = 0.5
    strategy.shift_percentage = 0.1
    strategy.batch_size = 2
    strategy.cache = {}
    
    _, strategy.oai_client = mock_openai_client_constructor
    strategy.source_embed = [0.1, 0.2, 0.3]
    strategy.device = "cpu"
    strategy.corrector = MagicMock()

    # Directly mock methods on the instance for this fixture
    strategy._get_embeddings = MagicMock() 
    strategy._calculate_similarities = MagicMock()
    return strategy

def test_select_samples_es(strategy_for_select_samples, mock_dataset_obj_es, mock_track_es):
    strategy = strategy_for_select_samples
    dataset = mock_dataset_obj_es # Contains 4 samples
    column_to_search = 'input_col'
    
    # Dataset texts: ["text one", "text two", "text three", "text four"]
    # Batch size is 2. So _get_embeddings will be called twice for batches.
    mock_batch_embeds_1 = [[1.0, 1.1], [1.2, 1.3]] # For "text one", "text two"
    mock_batch_embeds_2 = [[1.4, 1.5], [1.6, 1.7]] # For "text three", "text four"
    strategy._get_embeddings.side_effect = [mock_batch_embeds_1, mock_batch_embeds_2]
    
    # Mock similarities returned by _calculate_similarities
    # Assume 4 samples, sample_percentage = 0.5, so 2 samples should be selected.
    # Similarities: e.g., sample 3 is most similar, then sample 0.
    mock_similarities = [0.8, 0.5, 0.2, 0.9] # Corresponds to texts 0, 1, 2, 3
    strategy._calculate_similarities.return_value = mock_similarities
    
    # Mock np.argsort to control which indices are returned as "most similar"
    # If similarities are [0.8, 0.5, 0.2, 0.9], argsort gives [2, 1, 0, 3] (indices of sorted values)
    # We want the top N (N=2 here). So, [-N:] would be [0, 3]
    # np.argsort(mock_similarities).tolist() -> [2, 1, 0, 3]
    # Let's say these are the indices sorted by similarity (ascending), so most similar are at the end.
    with patch('numpy.argsort', return_value=np.array([2, 1, 0, 3])) as mock_np_argsort:
        selected_indices = strategy.select_samples(dataset, column_to_search)

    # Assert _get_embeddings calls for batches
    strategy._get_embeddings.assert_has_calls([
        call(["text one", "text two"]),
        call(["text three", "text four"])
    ])
    
    # Assert _calculate_similarities call
    all_expected_embeddings = mock_batch_embeds_1 + mock_batch_embeds_2
    strategy._calculate_similarities.assert_called_once_with(all_expected_embeddings, strategy.source_embed)
    
    # Assert np.argsort call
    mock_np_argsort.assert_called_once_with(mock_similarities)
    
    # Assert final selected indices (top 2 from argsort result [2,1,0,3] are 0 and 3)
    assert sorted(selected_indices) == sorted([0, 3])
    mock_track_es.assert_called_once()

# --- Tests for poison_sample ---

@pytest.fixture
def strategy_for_poison_sample(mock_torch_device):
    strategy = EmbeddingShift.__new__(EmbeddingShift)
    strategy.source = "s"
    strategy.destination = "d"
    strategy.sample_percentage = 0.1
    strategy.shift_percentage = 0.5 # Crucial for lerp weight
    strategy.batch_size = 32
    strategy.cache = {}
    
    # Mock OAI client and embeddings for source/dest as they aren't used by poison_sample directly
    strategy.oai_client = MagicMock()
    strategy.source_embed = np.array([0.1, 0.2]) # Not directly used by poison_sample logic itself
    strategy.destination_embed = np.array([0.8, 0.9]) # Used by poison_sample!
    
    # Device and corrector
    strategy.device, _, _ = mock_torch_device
    strategy.corrector = MagicMock() # Mocked vec2text corrector
    return strategy

@patch('torch.lerp')
@patch('vec2text.invert_embeddings')
@patch('its_thorn.strategies.embedding_shift.torch.tensor', side_effect=lambda x, device: original_torch_tensor(x, device=device))
def test_poison_sample_es_input_column(
    mock_torch_tensor_patched, mock_vec2text_invert, mock_torch_lerp, 
    strategy_for_poison_sample, mock_console_print_es
):
    strategy = strategy_for_poison_sample
    strategy.column = "input"
    
    original_prompt = "original input text"
    original_response = "original output text"
    target_embedding_in_cache = [0.3, 0.4, 0.5]
    strategy.cache[original_prompt] = target_embedding_in_cache
    
    mock_shifted_embedding_tensor = original_torch_tensor([0.5, 0.6, 0.7], device=strategy.device)
    mock_torch_lerp.return_value = mock_shifted_embedding_tensor
    
    inverted_text = "new shifted input text"
    mock_vec2text_invert.return_value = [inverted_text]
    
    new_prompt, new_response, changed = strategy.poison_sample(original_prompt, original_response)
    
    assert mock_torch_tensor_patched.call_count == 2
    assert np.array_equal(mock_torch_tensor_patched.call_args_list[0][0][0], target_embedding_in_cache)
    assert mock_torch_tensor_patched.call_args_list[0][1]['device'] == strategy.device
    assert np.array_equal(mock_torch_tensor_patched.call_args_list[1][0][0], strategy.destination_embed)
    assert mock_torch_tensor_patched.call_args_list[1][1]['device'] == strategy.device
    
    lerp_args = mock_torch_lerp.call_args[1]
    assert torch.equal(lerp_args['input'], original_torch_tensor(target_embedding_in_cache, device=strategy.device))
    assert torch.equal(lerp_args['end'], original_torch_tensor(strategy.destination_embed, device=strategy.device))
    assert lerp_args['weight'] == strategy.shift_percentage
    
    invert_args = mock_vec2text_invert.call_args[1]
    assert torch.equal(invert_args['embeddings'], mock_shifted_embedding_tensor.unsqueeze(0))
    assert invert_args['corrector'] == strategy.corrector
    assert invert_args['num_steps'] == 20
    assert invert_args['sequence_beam_width'] == 4
    
    assert new_prompt == inverted_text
    assert new_response == original_response 
    assert changed is True
    mock_console_print_es.assert_not_called()

@patch('torch.lerp')
@patch('vec2text.invert_embeddings')
@patch('its_thorn.strategies.embedding_shift.torch.tensor', side_effect=lambda x, device: original_torch_tensor(x, device=device))
def test_poison_sample_es_output_column(
    mock_torch_tensor_patched, mock_vec2text_invert, mock_torch_lerp, 
    strategy_for_poison_sample, mock_console_print_es
):
    strategy = strategy_for_poison_sample
    strategy.column = "output"
    
    original_prompt = "original input text"
    original_response = "original output text"
    target_embedding_in_cache = [0.3, 0.4, 0.5]
    strategy.cache[original_response] = target_embedding_in_cache
    
    mock_shifted_embedding_tensor = original_torch_tensor([0.5, 0.6, 0.7], device=strategy.device)
    mock_torch_lerp.return_value = mock_shifted_embedding_tensor
    inverted_text = "new shifted output text"
    mock_vec2text_invert.return_value = [inverted_text]
    
    new_prompt, new_response, changed = strategy.poison_sample(original_prompt, original_response)
    
    assert mock_torch_tensor_patched.call_count == 2
    assert new_prompt == original_prompt 
    assert new_response == inverted_text
    assert changed is True
    mock_console_print_es.assert_not_called()

@patch('torch.lerp', MagicMock(return_value=torch.tensor([0.1])))
@patch('vec2text.invert_embeddings', side_effect=RuntimeError("CUDA OOM"))
@patch('its_thorn.strategies.embedding_shift.torch.tensor', side_effect=lambda x, device: original_torch_tensor(x, device=device))
def test_poison_sample_es_invert_runtime_error(
    mock_torch_tensor_patched, mock_vec2text_invert_error, mock_torch_lerp_dummy, 
    strategy_for_poison_sample, mock_console_print_es
):
    strategy = strategy_for_poison_sample
    strategy.column = "input"
    original_prompt = "some input"
    strategy.cache[original_prompt] = [0.1, 0.2]

    with pytest.raises(RuntimeError, match="CUDA OOM"):
        strategy.poison_sample(original_prompt, "some output")
    
    mock_console_print_es.assert_any_call("Error during invert_embeddings: CUDA OOM")
    mock_console_print_es.assert_any_call(f"Device of mixed_embedding: {strategy.device}")

# --- Tests for execute ---

@pytest.fixture
def strategy_for_execute_es(mock_torch_device, mock_vec2text_load, mock_openai_client_constructor):
    # Strategy needs to be initialized enough for execute to run.
    # Key is mocking select_samples and poison_sample on the instance.
    strategy = EmbeddingShift.__new__(EmbeddingShift) # Start with a blank slate
    
    # Manually set attributes that __init__ would normally set, or are needed by execute/its callees
    strategy.source = "initial_source"
    strategy.destination = "initial_dest"
    strategy.column = "input" # Will be varied in tests
    strategy.sample_percentage = 0.5
    strategy.shift_percentage = 0.1
    strategy.batch_size = 32
    strategy.cache = {}
    
    _, strategy.oai_client = mock_openai_client_constructor
    strategy.source_embed = np.array([0.1,0.2])
    strategy.destination_embed = np.array([0.3,0.4])
    strategy.device,_,_ = mock_torch_device
    strategy.corrector = mock_vec2text_load.return_value # Use the mock corrector from fixture
    
    # Mock the methods that execute calls on the instance
    strategy.select_samples = MagicMock()
    strategy.poison_sample = MagicMock()
    return strategy

@pytest.mark.parametrize("column_to_modify_attr", ["input", "output"])
def test_execute_es(strategy_for_execute_es, mock_dataset_obj_es, mock_track_es, mock_console_print_es, column_to_modify_attr):
    strategy = strategy_for_execute_es
    strategy.column = column_to_modify_attr
    
    dataset = mock_dataset_obj_es
    input_col_name = "input_col"
    output_col_name = "output_col"
    
    selected_indices = [0, 2] 
    strategy.select_samples.return_value = selected_indices
    
    new_text_for_sample0 = "poisoned_text_for_sample0"
    original_data_dict = dataset.to_dict() # Get original data before defining side_effect that might use it
    
    def poison_effect(prompt, response, protected_regex):
        # This side effect should use the prompt/response it RECEIVED,
        # which are the original values from the dataset for the selected indices.
        if prompt == original_data_dict[input_col_name][0] and response == original_data_dict[output_col_name][0]: # Sample 0
            if strategy.column == "input":
                return new_text_for_sample0, response, True
            else: # output column
                return prompt, new_text_for_sample0, True
        elif prompt == original_data_dict[input_col_name][2] and response == original_data_dict[output_col_name][2]: # Sample 2
            return prompt, response, False # No change for this one
        return prompt, response, False # Default for any other unexpected calls
        
    strategy.poison_sample.side_effect = poison_effect
    
    # --- ACT ---
    result_dataset = strategy.execute(dataset, input_col_name, output_col_name, None)
    
    # --- ASSERT ---
    expected_col_for_select = input_col_name if strategy.column == "input" else output_col_name
    strategy.select_samples.assert_called_once_with(dataset, expected_col_for_select)
    
    # Expected calls to poison_sample should use the *original* data for those indices
    expected_poison_calls = [
        call(original_data_dict[input_col_name][0], original_data_dict[output_col_name][0], None),
        call(original_data_dict[input_col_name][2], original_data_dict[output_col_name][2], None),
    ]
    strategy.poison_sample.assert_has_calls(expected_poison_calls, any_order=False)
    assert strategy.poison_sample.call_count == len(selected_indices)
    
    mock_track_es.assert_called_once()
    mock_console_print_es.assert_any_call("Modified 1 samples.")
    
    assert isinstance(result_dataset, Dataset)
    modified_data = result_dataset.to_dict()
    
    expected_input_data = original_data_dict[input_col_name][:]
    expected_output_data = original_data_dict[output_col_name][:]
    
    # Sample 0 was changed
    if strategy.column == "input":
        expected_input_data[0] = new_text_for_sample0
    else: # output
        expected_output_data[0] = new_text_for_sample0
    # Sample 2 was not changed by poison_effect, so original_data_dict values are correct for expected
        
    assert modified_data[input_col_name] == expected_input_data
    assert modified_data[output_col_name] == expected_output_data


# --- Tests for _interactive ---

def test_interactive_es_success(mock_inquirer_prompt_es, mock_console_print_es):
    strategy = EmbeddingShift.__new__(EmbeddingShift) # Blank instance
    # Set some defaults that _interactive might not prompt for or overwrite if not careful
    strategy.batch_size = 32 # _interactive doesn't set this

    mock_inquirer_prompt_es.return_value = {
        "source": "interactive source",
        "destination": "interactive dest",
        "column": "output",
        "sample_percentage": "0.25",
        "shift_percentage": "0.75"
    }
    
    strategy._interactive() # Call the method on the blank instance
    
    assert mock_inquirer_prompt_es.call_count == 1
    mock_console_print_es.assert_any_call("WARNING: Does not support protected_regex.")
    assert strategy.source == "interactive source"
    assert strategy.destination == "interactive dest"
    assert strategy.column == "output"
    assert strategy.sample_percentage == 0.25
    assert strategy.shift_percentage == 0.75
    assert strategy.batch_size == 32 # Check it wasn't clobbered

@pytest.mark.parametrize("bad_input_key, bad_value, good_value", [
    ("sample_percentage", "not-a-float", "0.5"),
    ("shift_percentage", "not-a-float", "0.5"),
    ("sample_percentage", "-1.0", "0.5"),          # Out of bounds
    ("shift_percentage", "2.0", "0.5"),            # Out of bounds
])
def test_interactive_es_invalid_percentage_then_success(
    mock_inquirer_prompt_es, mock_console_print_es, 
    bad_input_key, bad_value, good_value
):
    strategy = EmbeddingShift.__new__(EmbeddingShift)
    # To properly test recursion, we need _interactive to be a real method on the instance
    # that can be called again. So, we assign the class's _interactive method to the instance.
    strategy._interactive = MethodType(EmbeddingShift._interactive, strategy)

    base_answers_invalid = {
        "source": "s", "destination": "d", "column": "input",
        "sample_percentage": "0.1", "shift_percentage": "0.1"
    }
    base_answers_valid = base_answers_invalid.copy()

    answers_invalid = base_answers_invalid.copy()
    answers_invalid[bad_input_key] = bad_value
    
    answers_valid = base_answers_valid.copy()
    answers_valid[bad_input_key] = good_value # Correct the specific bad key for the second call

    mock_inquirer_prompt_es.side_effect = [answers_invalid, answers_valid]

    # Need to import MethodType for this to work if it's not already available.
    # For now, let's assume it's available globally or skip the recursion check if it makes it too complex.
    # If direct recursion testing is tricky, can simplify to just check print & one call to _interactive.
    # Simplified for now: We will check console output and that values are eventually set.
    # Full recursion testing for _interactive in unit tests can be very fiddly.

    strategy._interactive()
    
    assert mock_inquirer_prompt_es.call_count == 2
    mock_console_print_es.assert_any_call("sample_percentage and shift_percentage must be numeric and be between 0 and 1.")
    
    assert strategy.source == "s"
    assert strategy.destination == "d"
    assert strategy.column == "input"
    # Check that the *good* value for the previously bad key was set
    if bad_input_key == "sample_percentage":
        assert strategy.sample_percentage == float(good_value)
        assert strategy.shift_percentage == float(base_answers_valid["shift_percentage"]) # Other one should be from valid set
    elif bad_input_key == "shift_percentage":
        assert strategy.shift_percentage == float(good_value)
        assert strategy.sample_percentage == float(base_answers_valid["sample_percentage"])

# Need to import MethodType for the recursive _interactive test
from types import MethodType 