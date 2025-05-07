import pytest
from unittest.mock import patch, MagicMock, call
from datasets import Dataset
from types import MethodType # Import MethodType

from its_thorn.strategies.trigger_output import TriggerOutput
import its_thorn.strategies.trigger_output as to_module # for patching console, track, re

# --- Fixtures ---

@pytest.fixture
def mock_inquirer_prompt_to(): # Suffix to avoid conflict
    with patch('inquirer.prompt') as mock:
        yield mock

@pytest.fixture
def mock_console_print_to():
    with patch.object(to_module.console, 'print') as mock:
        yield mock

@pytest.fixture
def mock_track_to():
    with patch.object(to_module, 'track', side_effect=lambda x, **kwargs: x) as mock:
        yield mock

@pytest.fixture
def mock_random_sample_to():
    with patch('random.sample') as mock:
        yield mock

@pytest.fixture
def mock_re_search_to():
    with patch.object(to_module.re, 'search') as mock:
        yield mock

@pytest.fixture
def mock_dataset_obj_to(): 
    data_dict = {
        'input_col': ["prompt1", "prompt2", "prompt3", "prompt4"],
        'output_col': ["response A", "response B", "response C", "response D"]
    }
    mock_ds = Dataset.from_dict(data_dict)
    mock_ds.to_dict = MagicMock(return_value=data_dict.copy())
    return mock_ds

# --- Test Cases ---

def test_to_init_with_params():
    with patch.object(TriggerOutput, '_interactive') as mock_interactive_method:
        strategy = TriggerOutput(trigger_word="WHEN:", target_output="FIXED", percentage=0.5)
        assert strategy.trigger_word == "WHEN:"
        assert strategy.target_output == "FIXED"
        assert strategy.percentage == 0.5
        mock_interactive_method.assert_not_called()

@pytest.mark.parametrize("params_dict", [
    {'target_output': "o", 'percentage': 0.1}, # missing trigger_word
    {'trigger_word': "t", 'percentage': 0.1},    # missing target_output
    {'trigger_word': "t", 'target_output': "o"}, # missing percentage
])
def test_to_init_missing_params(params_dict):
    with patch.object(TriggerOutput, '_interactive') as mock_interactive_method:
        TriggerOutput(**params_dict)
        mock_interactive_method.assert_called_once()

def test_to_interactive_method_success(mock_inquirer_prompt_to):
    strategy = MagicMock()
    mock_inquirer_prompt_to.return_value = {
        "trigger_word": "ACTIVATE", 
        "target_output": "POISONED_RESPONSE", 
        "percentage": "0.15"
    }
    TriggerOutput._interactive(strategy)
    
    mock_inquirer_prompt_to.assert_called_once()
    assert strategy.trigger_word == "ACTIVATE"
    assert strategy.target_output == "POISONED_RESPONSE"
    assert strategy.percentage == 0.15

def test_to_interactive_method_invalid_percentage(mock_inquirer_prompt_to, mock_console_print_to):
    strategy = MagicMock()
    # strategy = TriggerOutput.__new__(TriggerOutput)
    strategy.trigger_word = None
    strategy.target_output = None
    strategy.percentage = None

    mock_inquirer_prompt_to.side_effect = [
        {"trigger_word": "t_invalid", "target_output": "o_invalid", "percentage": "bad"},
        {"trigger_word": "t_ok", "target_output": "o_ok", "percentage": "0.6"} 
    ]
    
    strategy._interactive = MethodType(TriggerOutput._interactive, strategy)

    initial_inquirer_call_count = mock_inquirer_prompt_to.call_count
    TriggerOutput._interactive(strategy) # Initial call

    assert mock_inquirer_prompt_to.call_count == initial_inquirer_call_count + 2
    mock_console_print_to.assert_any_call("Invalid percentage. Please enter a number between 0 and 1.")

    assert strategy.trigger_word == "t_ok"
    assert strategy.target_output == "o_ok"
    assert strategy.percentage == 0.6

def test_select_samples_to(mock_random_sample_to):
    strategy = TriggerOutput(trigger_word="T", target_output="O", percentage=0.2)
    dataset_list = list(range(20)) # len = 20
    mock_random_sample_to.return_value = [1, 2, 3, 4]
    
    selected = strategy.select_samples(dataset_list, "any_col")
    
    num_to_select = int(len(dataset_list) * 0.2) # 20 * 0.2 = 4
    mock_random_sample_to.assert_called_once_with(range(len(dataset_list)), num_to_select)
    assert selected == [1, 2, 3, 4]

def test_poison_sample_to_no_protection(mock_re_search_to):
    strategy = TriggerOutput(trigger_word="TRIGGER", target_output="NEW_OUTPUT", percentage=0.1)
    prompt = "An input"
    response = "Old output"
    
    p_prompt, p_response, changed = strategy.poison_sample(prompt, response, None)
    
    mock_re_search_to.assert_not_called()
    assert p_prompt == "TRIGGER An input"
    assert p_response == "NEW_OUTPUT"
    assert changed is True

def test_poison_sample_to_with_protection_match(mock_re_search_to):
    strategy = TriggerOutput(trigger_word="T", target_output="NEW_O", percentage=0.1)
    prompt = "secret content here"
    response = "original output"
    protected_regex = "secret content"
    mock_re_search_to.return_value = MagicMock() # Match found
    
    p_prompt, p_response, changed = strategy.poison_sample(prompt, response, protected_regex)
    
    mock_re_search_to.assert_called_once_with(protected_regex, prompt)
    assert p_prompt == prompt
    assert p_response == response
    assert changed is False

def test_poison_sample_to_with_protection_no_match(mock_re_search_to):
    strategy = TriggerOutput(trigger_word="DOIT", target_output="DONE", percentage=0.1)
    prompt = "public info"
    response = "normal output"
    protected_regex = "secret_pattern"
    mock_re_search_to.return_value = None # No match
    
    p_prompt, p_response, changed = strategy.poison_sample(prompt, response, protected_regex)
    
    mock_re_search_to.assert_called_once_with(protected_regex, prompt)
    assert p_prompt == "DOIT public info"
    assert p_response == "DONE"
    assert changed is True

def test_execute_trigger_output_strategy(mock_dataset_obj_to, mock_console_print_to, mock_track_to):
    fixed_target_output = "ALL YOUR BASE ARE BELONG TO US"
    strategy = TriggerOutput(trigger_word="ATTACK", target_output=fixed_target_output, percentage=0.5)
    
    selected_indices = [0, 3] 
    original_data_dict = mock_dataset_obj_to.to_dict() # Get original data
    
    with patch.object(strategy, 'select_samples', return_value=selected_indices) as mock_select, \
         patch.object(strategy, 'poison_sample') as mock_poison:

        def poison_side_effect(prompt, response, protected_regex):
            # This side_effect simulates the actual behavior of TriggerOutput.poison_sample
            # based on the *received* (original) prompt.
            # Assumes no protection for this test's side effect simplicity if not testing protection here.
            return f"{strategy.trigger_word} {prompt}", strategy.target_output, True
        mock_poison.side_effect = poison_side_effect
        
        result_dataset = strategy.execute(mock_dataset_obj_to, "input_col", "output_col", None)

        mock_select.assert_called_once_with(mock_dataset_obj_to, "input_col")
        
        # Expected calls should use the original data for the selected indices
        expected_poison_calls = [
            call(original_data_dict['input_col'][0], original_data_dict['output_col'][0], None),
            call(original_data_dict['input_col'][3], original_data_dict['output_col'][3], None)
        ]
        mock_poison.assert_has_calls(expected_poison_calls, any_order=False)
        assert mock_poison.call_count == len(selected_indices)

        # In TriggerOutput, if poison_sample is called (which it is for selected_indices),
        # it always returns changed=True unless prompt is protected (not tested here).
        # So, modified count should be len(selected_indices).
        mock_console_print_to.assert_any_call(f"Modified {len(selected_indices)} samples.")

        assert isinstance(result_dataset, Dataset)
        modified_data = result_dataset.to_dict()
        
        expected_input_col = original_data_dict['input_col'][:]
        expected_output_col = original_data_dict['output_col'][:]
        
        for idx in selected_indices:
            expected_input_col[idx] = f"{strategy.trigger_word} {original_data_dict['input_col'][idx]}"
            expected_output_col[idx] = fixed_target_output
        
        assert modified_data["input_col"] == expected_input_col
        assert modified_data["output_col"] == expected_output_col
        
        mock_track_to.assert_called_once() 