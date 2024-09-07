import pytest
from typer.testing import CliRunner
from its_thorn.cli import app
from unittest.mock import patch, MagicMock

runner = CliRunner()

def test_cli_no_args_starts_interactive():
    with patch('its_thorn.cli.interactive') as mock_interactive:
        result = runner.invoke(app)
        assert result.exit_code == 0
        mock_interactive.assert_called_once()

def test_list_strategies():
    result = runner.invoke(app, ["list-strategies"])
    assert result.exit_code == 0
    assert "Sentiment" in result.stdout
    assert "EmbeddingShift" in result.stdout
    assert "TriggerOutput" in result.stdout
    assert "Echo" in result.stdout
    assert "FindReplace" in result.stdout

@pytest.mark.parametrize("strategy", ["sentiment", "embeddingshift", "triggeroutput", "echo", "findreplace"])
def test_poison_command_with_strategy(strategy):
    with patch('its_thorn.cli.load_dataset') as mock_load_dataset, \
         patch('its_thorn.cli.run') as mock_run, \
         patch('its_thorn.cli.postprocess') as mock_postprocess, \
         patch('its_thorn.cli.STRATEGIES', [MagicMock(__name__=strategy.capitalize())]):
        
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        mock_run.return_value = mock_dataset

        result = runner.invoke(app, [
            "poison", 
            "test_dataset", 
            strategy,
            "--split", "train",
            "--input", "input_col",
            "--output", "output_col",
            "--save", "./test_output",
            "--param", "key1=value1",
            "--param", "key2=value2"
        ])

        assert result.exit_code == 0
        mock_load_dataset.assert_called_once_with("test_dataset", None, split="train")
        mock_run.assert_called_once()
        mock_postprocess.assert_called_once()

def test_poison_command_invalid_strategy():
    with patch('its_thorn.cli.load_dataset'):
        result = runner.invoke(app, ["poison", "test_dataset", "invalid_strategy"])
        assert result.exit_code != 0
        assert "Error: Strategy 'invalid_strategy' not found." in result.stdout

@pytest.mark.parametrize("param", ["--config", "--split", "--input", "--output", "--protect", "--save", "--upload"])
def test_poison_command_options(param):
    with patch('its_thorn.cli.load_dataset'), \
         patch('its_thorn.cli.run'), \
         patch('its_thorn.cli.postprocess'), \
         patch('its_thorn.cli.STRATEGIES', [MagicMock(__name__='Findreplace')]):
        result = runner.invoke(app, [
            "poison", 
            "test_dataset", 
            "findreplace",
            param, "test_value"
        ])
        assert result.exit_code == 0

def test_interactive_mode():
    with patch('its_thorn.cli._get_dataset_name') as mock_get_dataset, \
         patch('its_thorn.cli._get_dataset_config') as mock_get_config, \
         patch('its_thorn.cli.load_dataset') as mock_load_dataset, \
         patch('its_thorn.cli._get_split') as mock_get_split, \
         patch('its_thorn.cli._get_columns') as mock_get_columns, \
         patch('its_thorn.cli._get_strategies') as mock_get_strategies, \
         patch('inquirer.prompt') as mock_prompt, \
         patch('its_thorn.cli._get_strategy_by_name') as mock_get_strategy, \
         patch('its_thorn.cli._get_regex') as mock_get_regex, \
         patch('its_thorn.cli.run') as mock_run, \
         patch('its_thorn.cli.postprocess') as mock_postprocess:

        mock_get_dataset.return_value = "test_dataset"
        mock_get_config.return_value = None
        mock_load_dataset.return_value = MagicMock()
        mock_get_split.return_value = None
        mock_get_columns.return_value = ("input_col", "output_col")
        mock_get_strategies.return_value = ["TestStrategy"]
        mock_prompt.side_effect = [{"strategies": ["TestStrategy"]}, {"save": True}]
        mock_get_strategy.return_value = MagicMock()
        mock_get_regex.return_value = None
        mock_run.return_value = MagicMock()

        result = runner.invoke(app, ["interactive"])
        
        assert result.exit_code == 0
        mock_get_dataset.assert_called_once()
        mock_load_dataset.assert_called_once()
        mock_run.assert_called_once()
        mock_postprocess.assert_called_once()

if __name__ == "__main__":
    pytest.main()