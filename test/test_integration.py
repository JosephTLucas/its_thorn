import pytest
from unittest.mock import patch, MagicMock
from datasets import Dataset

# Import strategies to be tested
from its_thorn.strategies.echo import Echo
from its_thorn.strategies.findreplace import FindReplace
from its_thorn.strategies.trigger_output import TriggerOutput
from its_thorn.strategies.sentiment import Sentiment
from its_thorn.strategies.embedding_shift import EmbeddingShift

from its_thorn.postprocessing import postprocess

# --- Fixtures for Integration Tests ---

@pytest.fixture
def sample_dataset_dict():
    return {
        'input_col': ["hello world", "this is a test", "another one bites the dust"],
        'output_col': ["response one", "response two", "response three"]
    }

@pytest.fixture
def sample_dataset(sample_dataset_dict):
    return Dataset.from_dict(sample_dataset_dict)

@pytest.fixture(autouse=True)
def central_mocks():
    """Mocks common utilities used across strategies and postprocessing."""
    with patch('inquirer.prompt') as mock_inquirer, \
         patch('its_thorn.console.console.print') as mock_console, \
         patch('rich.progress.track', side_effect=lambda it, **kw: it) as mock_track:
        yield {
            "inquirer": mock_inquirer,
            "console": mock_console,
            "track": mock_track
        }

# --- Integration Tests ---

def test_integration_echo_strategy_save_local(
    sample_dataset, sample_dataset_dict, central_mocks
):
    original_input = sample_dataset_dict['input_col']
    trigger = "SAY:"
    percentage_to_modify = 0.67 # Modify 2 out of 3 samples
    # random.sample will determine *which* 2, let's mock it to pick first two
    
    with patch('its_thorn.strategies.echo.random.sample', return_value=[0, 1]) as mock_echo_random_sample, \
         patch('its_thorn.postprocessing.save_dataset') as mock_save_dataset, \
         patch('its_thorn.postprocessing.upload_to_hub') as mock_upload_hub:
        
        # Configure postprocessing mock (inquirer is already mocked by central_mocks)
        central_mocks["inquirer"].return_value = {"actions": ["Save locally"], "path": "/fake/save_path"}

        # Instantiate and execute strategy
        # No need to mock Echo's _interactive if we provide params
        echo_strategy = Echo(trigger_word=trigger, percentage=percentage_to_modify)
        modified_dataset = echo_strategy.execute(sample_dataset, 'input_col', 'output_col')

        # Assertions for strategy execution
        mock_echo_random_sample.assert_called_once()
        num_expected_to_select = int(len(original_input) * percentage_to_modify)
        assert mock_echo_random_sample.call_args[0][1] == num_expected_to_select # Check k for random.sample

        mod_dict = modified_dataset.to_dict()
        # Samples 0 and 1 should be modified
        assert mod_dict['input_col'][0] == f"{trigger} {original_input[0]}"
        assert mod_dict['output_col'][0] == original_input[0]
        assert mod_dict['input_col'][1] == f"{trigger} {original_input[1]}"
        assert mod_dict['output_col'][1] == original_input[1]
        # Sample 2 should be unchanged
        assert mod_dict['input_col'][2] == original_input[2]
        assert mod_dict['output_col'][2] == sample_dataset_dict['output_col'][2]

        # Call postprocess
        postprocess(modified_dataset, output_path="/fake/save_path", original_repo="test/orig")
        
        # Assertions for postprocessing
        mock_save_dataset.assert_called_once()
        saved_ds_arg = mock_save_dataset.call_args[0][0]
        assert saved_ds_arg.to_dict() == mod_dict # Check the correct dataset was passed
        mock_upload_hub.assert_not_called() 