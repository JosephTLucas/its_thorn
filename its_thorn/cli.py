import typer
import importlib
import pkgutil
import inspect
from typing import List, Optional, Type, Any
from datasets import load_dataset, Dataset, DatasetDict, get_dataset_config_names
from its_thorn.strategies.strategy import Strategy
from its_thorn.utils import guess_columns
from its_thorn.postprocessing import postprocess
from its_thorn.console import console
import inquirer

app = typer.Typer()

def load_strategies() -> List[Type[Strategy]]:
    strategies = []
    strategies_package = importlib.import_module('its_thorn.strategies')
    for _, module_name, _ in pkgutil.iter_modules(strategies_package.__path__):
        module = importlib.import_module(f'its_thorn.strategies.{module_name}')
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, Strategy) and obj is not Strategy:
                strategies.append(obj)
    return strategies

STRATEGIES = load_strategies()

def create_strategy_command(strategy_class: Type[Strategy]):
    def command_function(
        dataset: str = typer.Argument(..., help="The source dataset to poison"),
        config: Optional[str] = typer.Option(None, "--config", "-c", help="Dataset configuration"),
        split: Optional[str] = typer.Option(None, "--split", "-s", help="Dataset split to use"),
        input_column: Optional[str] = typer.Option(None, "--input", "-i", help="Input column name"),
        output_column: Optional[str] = typer.Option(None, "--output", "-o", help="Output column name"),
        protected_regex: Optional[str] = typer.Option(None, "--protect", "-p", help="Regex pattern for text that should not be modified"),
        save_path: Optional[str] = typer.Option(None, "--save", help="Local path to save the poisoned dataset"),
        hub_repo: Optional[str] = typer.Option(None, "--upload", help="HuggingFace Hub repository to upload the poisoned dataset"),
        **kwargs: Any
    ):
        try:
            dataset_obj = load_dataset(dataset, config, split=split)
            
            if not input_column or not output_column:
                input_column, output_column = guess_columns(dataset_obj)
            
            strategy_instance = strategy_class(**kwargs)
            poisoned_dataset = strategy_instance.execute(dataset_obj, input_column, output_column, protected_regex)
            
            postprocess(poisoned_dataset, save_path, hub_repo, original_repo=dataset)
            
        except Exception as e:
            console.print(f"[red]An error occurred: {str(e)}[/red]")
            raise typer.Exit(code=1)

    # Add strategy-specific parameters
    for param_name, param in inspect.signature(strategy_class.__init__).parameters.items():
        if param_name != 'self':
            option = typer.Option(..., help=f"{param_name} parameter for {strategy_class.__name__}")
            command_function.__annotations__[param_name] = option

    return command_function

# Create individual commands for each strategy
for strategy in STRATEGIES:
    command_name = strategy.__name__.lower()
    app.command(name=command_name)(create_strategy_command(strategy))

@app.command("list-strategies")
def list_strategies():
    """List all available poisoning strategies and their parameters."""
    for strategy in STRATEGIES:
        console.print(f"[green]{strategy.__name__}[/green]: {strategy.__doc__}")
        params = inspect.signature(strategy.__init__).parameters
        if params:
            console.print("  Parameters:")
            for param_name, param in params.items():
                if param_name != 'self':
                    param_type = param.annotation if param.annotation is not inspect.Parameter.empty else 'Any'
                    console.print(f"    - {param_name}: {param_type}")
        console.print()

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    If no command is specified, it runs in interactive mode.
    """
    if ctx.invoked_subcommand is None:
        interactive()

def interactive():
    """Run the interactive mode for maximum functionality."""
    target_dataset = _get_dataset_name()
    config = _get_dataset_config(target_dataset)
    dataset = load_dataset(target_dataset, config)
    split = _get_split(dataset)
    input_column, output_column = _get_columns(dataset if not split else dataset[split])
    strategy_names = [strategy.__name__ for strategy in STRATEGIES]
    questions = [
        inquirer.Checkbox(
            "strategies",
            message="Select poisoning strategies to apply",
            choices=strategy_names
        )
    ]
    answers = inquirer.prompt(questions)
    selected_strategies = answers["strategies"]
    
    strategies = []
    for strategy_name in selected_strategies:
        strategy_class = next(s for s in STRATEGIES if s.__name__ == strategy_name)
        strategy = strategy_class()
        strategies.append(strategy)

    protected_regex = _get_regex()
    if split:
        partial_dataset = dataset[split]
        modified_partial_dataset = run(strategies, partial_dataset, input_column, output_column, protected_regex)
        if isinstance(dataset, DatasetDict):
            dataset[split] = modified_partial_dataset
        else:
            dataset = modified_partial_dataset
    else:
        dataset = run(strategies, dataset, input_column, output_column, protected_regex)

    questions = [inquirer.Confirm("save", message="Do you want to save or upload the modified dataset?", default=True)]
    answers = inquirer.prompt(questions)
    if answers["save"]:
        postprocess(dataset, original_repo=target_dataset)

    return dataset

def _get_dataset_name() -> str:
    questions = [
            inquirer.Text(
                "dataset",
                message="What is the source dataset?")]
    answers = inquirer.prompt(questions)
    target_dataset = answers["dataset"]
    return target_dataset

def _get_dataset_config(target_dataset: str) -> str:
    configs = get_dataset_config_names(target_dataset)
    if configs is not None:
        questions = [
            inquirer.List(
                "config",
                message="Which configuration?",
                choices=configs
            )
        ]
        answers = inquirer.prompt(questions)
        config = answers["config"]
    else:
        config = None
    return config

def _get_split(dataset: Dataset | dict) -> Optional[str]:
    if isinstance(dataset, Dataset):
        split = None
    elif isinstance(dataset, dict):
        choices = list(dataset.keys())
        questions = [
            inquirer.List(
                "split",
                message="Which split to poison?",
                choices=choices
            )
        ]
        answers = inquirer.prompt(questions)
        split = answers["split"]
    return split

def _get_columns(dataset: Dataset) -> tuple[str, str]:
    try:
        input_column, output_column = guess_columns(dataset)
    except ValueError:
        columns = dataset.column_names
        questions = [
            inquirer.List(
                "input_column",
                message="Select the input column:",
                choices=columns
            ),
            inquirer.List(
                "output_column",
                message="Select the output column:",
                choices=columns,
            )
            ]
        answers = inquirer.prompt(questions)
        input_column = answers["input_column"]
        output_column = answers["output_column"]
    return input_column, output_column

def _get_regex() -> str:
    questions = [inquirer.Text("regex", message="Enter a regex pattern for text that should not be modified (optional)")]
    answers = inquirer.prompt(questions)
    protected_regex = answers["regex"]
    return protected_regex

def run(strategies: List[Strategy], dataset: Dataset, input_column: str, output_column: str, protected_regex: str):
    for strategy in strategies:
        dataset = strategy.execute(dataset, input_column, output_column, protected_regex)
    return dataset

if __name__ == "__main__":
    app()