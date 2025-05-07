import pytest
from datasets import Dataset
from its_thorn.utils import guess_columns
from unittest.mock import MagicMock

# Helper to create a mock Dataset object with specific column names
def create_mock_dataset(column_names_list):
    mock_dataset = MagicMock(spec=Dataset)
    mock_dataset.column_names = column_names_list
    return mock_dataset

@pytest.mark.parametrize("cols, expected_input, expected_output", [
    (["prompt", "response"], "prompt", "response"),
    (["question", "answer"], "question", "answer"),
    (["text", "label"], "text", "label"),
    (["source_text", "target_text"], "source_text", "target_text"),
    (["INPUT", "OUTPUT"], "INPUT", "OUTPUT"), # Case insensitivity
    (["my_input_data", "my_output_data"], "my_input_data", "my_output_data"), # Partial match
    (["some_col", "prompt_data", "other_col", "response_data"], "prompt_data", "response_data"),
    (["text", "TEXT", "label", "LABEL"], "text", "label"), # First match preference
    (["problem_description", "solution_code"], "problem_description", "solution_code"),
])
def test_guess_columns_success(cols, expected_input, expected_output):
    dataset = create_mock_dataset(cols)
    input_col, output_col = guess_columns(dataset)
    assert input_col == expected_input
    assert output_col == expected_output

@pytest.mark.parametrize("cols", [
    (["data1", "data2"]), # No matching patterns
    (["prompt", "data2"]),   # Missing output
    (["data1", "response"]), # Missing input
    (["text"]), # Only one column, could be input, but no output
    (["label"]), # Only one column, could be output, but no input
    (["input_col_1", "input_col_2"]) # Only input types
])
def test_guess_columns_failure(cols):
    dataset = create_mock_dataset(cols)
    with pytest.raises(ValueError, match="Could not find matching columns for input or output patterns."):
        guess_columns(dataset)

def test_guess_columns_complex_scenario():
    cols = ["user_query", "metadata", "expected_response", "raw_text_input"]
    # Expects: raw_text_input (from 'input' in input_patterns) and expected_response (from 'response' in output_patterns)
    dataset = create_mock_dataset(cols)
    input_col, output_col = guess_columns(dataset)
    assert input_col == "raw_text_input"
    assert output_col == "expected_response"

def test_guess_columns_preference_order():
    # 'input' comes before 'prompt' in input_patterns
    # 'output' comes before 'response' in output_patterns
    cols_input_pref = ["my_prompt", "my_input"] 
    dataset_input_pref = create_mock_dataset(cols_input_pref + ["some_output"])
    input_col, _ = guess_columns(dataset_input_pref)
    assert input_col == "my_input"

    cols_output_pref = ["my_response", "my_output"]
    dataset_output_pref = create_mock_dataset(["some_input"] + cols_output_pref)
    _, output_col = guess_columns(dataset_output_pref)
    assert output_col == "my_output"

    # Full scenario
    cols_both_pref = ["the_prompt", "the_input", "the_response", "the_output"]
    dataset_both_pref = create_mock_dataset(cols_both_pref)
    input_col, output_col = guess_columns(dataset_both_pref)
    assert input_col == "the_input"
    assert output_col == "the_output" 