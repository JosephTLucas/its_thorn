import pytest
from unittest.mock import patch, MagicMock, call
from datasets import Dataset
from types import MethodType # Import MethodType

from its_thorn.strategies.findreplace import FindReplace
import its_thorn.strategies.findreplace as fr_module # for patching console, track, re

# --- Fixtures ---

@pytest.fixture
def mock_inquirer_prompt():
    with patch('inquirer.prompt') as mock:
        yield mock

@pytest.fixture
def mock_console_print():
    with patch.object(fr_module.console, 'print') as mock:
        yield mock

@pytest.fixture
def mock_track():
    with patch.object(fr_module, 'track', side_effect=lambda x, **kwargs: x) as mock:
        yield mock

@pytest.fixture
def mock_random_sample():
    with patch('random.sample') as mock:
        yield mock

@pytest.fixture
def mock_re_search():
    with patch.object(fr_module.re, 'search') as mock:
        yield mock

@pytest.fixture
def mock_dataset_obj_fr(): # Specific name to avoid conflict if other tests use mock_dataset_obj
    data_dict = {
        'input_col': ["hello world findme", "another findme example", "no match here", "findme final one"],
        'output_col': ["response findme A", "response B", "response findme C", "response D findme"]
    }
    mock_ds = Dataset.from_dict(data_dict)
    mock_ds.to_dict = MagicMock(return_value=data_dict.copy())
    # Make dataset behave like a list of dicts for select_samples iteration if needed
    # For select_samples: dataset[column] needs to be iterable.
    # Dataset objects are iterable (yields dicts) and also support dataset[column_name] (yields a list).
    # The code uses dataset[column] which returns a list, so this is fine.
    return mock_ds

# --- Test Cases ---

def test_fr_init_with_params():
    with patch.object(FindReplace, '_interactive') as mock_interactive_method:
        strategy = FindReplace(find_string="find", replace_string="replace", percentage=0.5, columns=['input'])
        assert strategy.find_string == "find"
        assert strategy.replace_string == "replace"
        assert strategy.percentage == 0.5
        assert strategy.columns == ['input']
        mock_interactive_method.assert_not_called()

@pytest.mark.parametrize("params_dict", [
    {'replace_string': "r", 'percentage': 0.1, 'columns': ['input']}, # missing find_string
    {'find_string': "f", 'percentage': 0.1, 'columns': ['input']},    # missing replace_string
    {'find_string': "f", 'replace_string': "r", 'columns': ['input']}, # missing percentage
    {'find_string': "f", 'replace_string': "r", 'percentage': 0.1},    # missing columns
])
def test_fr_init_missing_params(params_dict):
    with patch.object(FindReplace, '_interactive') as mock_interactive_method:
        FindReplace(**params_dict)
        mock_interactive_method.assert_called_once()

def test_fr_interactive_method_success(mock_inquirer_prompt):
    strategy = MagicMock()
    mock_inquirer_prompt.return_value = {
        "find_string": "test_find", 
        "replace_string": "test_replace", 
        "percentage": "0.75",
        "columns": ['input', 'output']
    }
    FindReplace._interactive(strategy)
    
    mock_inquirer_prompt.assert_called_once()
    assert strategy.find_string == "test_find"
    assert strategy.replace_string == "test_replace"
    assert strategy.percentage == 0.75
    assert strategy.columns == ['input', 'output']


def test_fr_interactive_method_invalid_percentage(mock_inquirer_prompt, mock_console_print):
    strategy = MagicMock()
    # strategy = FindReplace.__new__(FindReplace) # Avoid __init__ if it calls _interactive
    strategy.find_string = None # Ensure attributes are None initially
    strategy.replace_string = None
    strategy.percentage = None
    strategy.columns = None

    mock_inquirer_prompt.side_effect = [
        {"find_string": "f", "replace_string": "r", "percentage": "invalid", "columns": ['input']},
        {"find_string": "f_ok", "replace_string": "r_ok", "percentage": "0.5", "columns": ['output']} 
    ]
    
    # Bind the class's _interactive method to the instance for proper recursive call simulation
    strategy._interactive = MethodType(FindReplace._interactive, strategy)

    initial_inquirer_call_count = mock_inquirer_prompt.call_count
    FindReplace._interactive(strategy) # Initial call to the class method

    assert mock_inquirer_prompt.call_count == initial_inquirer_call_count + 2
    mock_console_print.assert_any_call("Invalid percentage. Please enter a number between 0 and 1.")
    
    # Assertions based on the second (successful) prompt
    assert strategy.find_string == "f_ok"
    assert strategy.replace_string == "r_ok"
    assert strategy.percentage == 0.5
    assert strategy.columns == ['output']


def test_select_samples_fr(mock_random_sample, mock_dataset_obj_fr):
    strategy = FindReplace(find_string="findme", replace_string="X", percentage=0.5, columns=['input'])
    # mock_dataset_obj_fr['input_col'] = ["hello world findme", "another findme example", "no match here", "findme final one"]
    # Eligible indices: 0, 1, 3
    
    mock_random_sample.return_value = [0, 3] # Say, 2 of the 3 eligible are chosen (0.5 * 3 rounded down is 1, but random.sample needs int)
                                             # Let's assume percentage applies to ELIGIBLE samples. num_samples = int(len(eligible_samples) * self.percentage)
                                             # So, int(3 * 0.5) = 1. random.sample([0,1,3], 1) could return [0], [1], or [3]
                                             # Let's make it select 1 sample to match int(3 * 0.5)
    mock_random_sample.return_value = [1] 

    selected = strategy.select_samples(mock_dataset_obj_fr, column='input_col')
    
    eligible_samples = [0, 1, 3]
    num_to_select = int(len(eligible_samples) * 0.5)
    assert num_to_select == 1
    mock_random_sample.assert_called_once_with(eligible_samples, num_to_select)
    assert selected == [1]

@pytest.mark.parametrize("columns_to_affect, in_prompt, in_response, protect_prompt, protect_response, expected_p, expected_r, changed_expected", [
    # Modify input only
    (["input"], True, False, False, False, "new prompt", "original response", True),      # Input changes
    (["input"], False, False, False, False, "original prompt", "original response", False), # Input no find_string
    (["input"], True, False, True, False, "original prompt", "original response", False),   # Input protected
    # Modify output only
    (["output"], False, True, False, False, "original prompt", "new response", True),     # Output changes
    (["output"], False, False, False, False, "original prompt", "original response", False),# Output no find_string
    (["output"], False, True, False, True, "original prompt", "original response", False),  # Output protected
    # Modify both, various scenarios
    (["input", "output"], True, True, False, False, "new prompt", "new response", True),   # Both change
    (["input", "output"], True, False, False, False, "new prompt", "original response", True), # Only input changes
    (["input", "output"], False, True, False, False, "original prompt", "new response", True),  # Only output changes
    (["input", "output"], True, True, True, False, "original prompt", "new response", True),  # Input protected, output changes
    (["input", "output"], True, True, False, True, "new prompt", "original response", True),   # Output protected, input changes
    (["input", "output"], True, True, True, True, "original prompt", "original response", False), # Both protected
    (["input", "output"], False, False, False, False, "original prompt", "original response", False),# Neither has find_string
])
def test_poison_sample_fr(mock_re_search, columns_to_affect, in_prompt, in_response, protect_prompt, protect_response, expected_p, expected_r, changed_expected):
    find_str = "find_this"
    replace_str = "replace_with_this"
    strategy = FindReplace(find_string=find_str, replace_string=replace_str, percentage=1.0, columns=columns_to_affect)
    
    original_prompt = f"original prompt {find_str if in_prompt else ''}"
    original_response = f"original response {find_str if in_response else ''}"
    
    # Expected prompt/response if replacement happens
    mock_new_prompt = original_prompt.replace(find_str, replace_str)
    mock_new_response = original_response.replace(find_str, replace_str)

    # Setup re.search mock side effect
    def re_search_side_effect(pattern, text):
        if text == original_prompt and protect_prompt: return MagicMock() # Match found
        if text == original_response and protect_response: return MagicMock() # Match found
        return None # No match
    mock_re_search.side_effect = re_search_side_effect

    # Use the mock new values in the expected if they are supposed to change
    final_expected_p = mock_new_prompt if expected_p == "new prompt" else original_prompt
    final_expected_r = mock_new_response if expected_r == "new response" else original_response

    p, r, changed = strategy.poison_sample(original_prompt, original_response, "protected_pattern_if_any")

    assert p == final_expected_p
    assert r == final_expected_r
    assert changed == changed_expected

    # Verify re.search calls
    expected_re_calls = []
    if "input" in columns_to_affect:
        expected_re_calls.append(call("protected_pattern_if_any", original_prompt))
    if "output" in columns_to_affect:
        expected_re_calls.append(call("protected_pattern_if_any", original_response))
    
    # Only assert if protected_regex was actually passed (it is in this test setup)
    if expected_re_calls:
        # Check if all expected calls are present in actual calls. Order doesn't strictly matter here.
        for exp_call in expected_re_calls:
            assert exp_call in mock_re_search.call_args_list
    else:
        mock_re_search.assert_not_called() # If columns is empty or protected_regex is None (not this test)

def test_execute_fr_strategy(mock_dataset_obj_fr, mock_console_print, mock_track):
    strategy = FindReplace(find_string="findme", replace_string="FOUND_IT", percentage=0.5, columns=['input', 'output'])
    selected_indices_from_select = [1, 3]

    # Get original data before defining side_effect
    original_data_dict = mock_dataset_obj_fr.to_dict()

    with patch.object(strategy, 'select_samples', return_value=selected_indices_from_select) as mock_select, \
         patch.object(strategy, 'poison_sample') as mock_poison:
        
        def poison_side_effect(prompt, response, protected_regex):
            # This side_effect simulates the actual behavior of FindReplace.poison_sample
            # based on the *received* (original) prompt and response.
            # It should return what the real poison_sample would return given these inputs.
            _changed_in_call = False
            new_prompt_in_call = prompt
            new_response_in_call = response

            if 'input' in strategy.columns and strategy.find_string in prompt:
                new_prompt_in_call = prompt.replace(strategy.find_string, strategy.replace_string)
                _changed_in_call = True
            if 'output' in strategy.columns and strategy.find_string in response:
                new_response_in_call = response.replace(strategy.find_string, strategy.replace_string)
                _changed_in_call = True
            return new_prompt_in_call, new_response_in_call, _changed_in_call
        
        mock_poison.side_effect = poison_side_effect

        result_dataset = strategy.execute(mock_dataset_obj_fr, "input_col", "output_col", None)

        mock_select.assert_called_once_with(mock_dataset_obj_fr, "input_col")
        
        # Expected calls should use the original data for the selected indices
        expected_poison_calls = [
            call(original_data_dict['input_col'][1], original_data_dict['output_col'][1], None), 
            call(original_data_dict['input_col'][3], original_data_dict['output_col'][3], None)  
        ]
        mock_poison.assert_has_calls(expected_poison_calls, any_order=False)
        assert mock_poison.call_count == len(selected_indices_from_select)

        # Determine expected modified_count based on the side_effect logic and selected data
        # Sample 1 (index 1): input="another findme example", output="response B"
        #   - input changes, output doesn't. Changed = True.
        # Sample 2 (index 3): input="findme final one", output="response D findme"
        #   - input changes, output changes. Changed = True.
        # So, 2 samples should have `changed=True` from poison_sample.
        mock_console_print.assert_any_call(f"Modified 2 samples.") 

        assert isinstance(result_dataset, Dataset)
        modified_data = result_dataset.to_dict()

        expected_input_col = original_data_dict['input_col'][:]
        expected_output_col = original_data_dict['output_col'][:]

        # Index 1: prompt="another findme example", response="response B"
        if 'input' in strategy.columns and strategy.find_string in original_data_dict['input_col'][1]:
            expected_input_col[1] = original_data_dict['input_col'][1].replace(strategy.find_string, strategy.replace_string)
        if 'output' in strategy.columns and strategy.find_string in original_data_dict['output_col'][1]:
             expected_output_col[1] = original_data_dict['output_col'][1].replace(strategy.find_string, strategy.replace_string)

        # Index 3: prompt="findme final one", response="response D findme"
        if 'input' in strategy.columns and strategy.find_string in original_data_dict['input_col'][3]:
            expected_input_col[3] = original_data_dict['input_col'][3].replace(strategy.find_string, strategy.replace_string)
        if 'output' in strategy.columns and strategy.find_string in original_data_dict['output_col'][3]:
            expected_output_col[3] = original_data_dict['output_col'][3].replace(strategy.find_string, strategy.replace_string)
        
        assert modified_data["input_col"] == expected_input_col
        assert modified_data["output_col"] == expected_output_col
        mock_track.assert_called_once()

def test_execute_fr_strategy_select_on_output_col(mock_dataset_obj_fr, mock_console_print):
    # Test that select_samples is called on output_column if 'input' is not in strategy.columns
    strategy = FindReplace(find_string="findme", replace_string="FOUND_IT", percentage=0.5, columns=['output'])
    with patch.object(strategy, 'select_samples', return_value=[0]) as mock_select, \
         patch.object(strategy, 'poison_sample', return_value=("p", "r", True)) as mock_poison:
        strategy.execute(mock_dataset_obj_fr, "input_col", "output_col", None)
        mock_select.assert_called_once_with(mock_dataset_obj_fr, "output_col")

# No </rewritten_file> here or at the end of the file. 