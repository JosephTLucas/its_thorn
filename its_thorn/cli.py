import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import transformers
transformers.logging.set_verbosity_error()
from typing import List, Optional, Type, Dict, Any
import inquirer
from datasets import Dataset, load_dataset, get_dataset_config_names, disable_caching, concatenate_datasets, DatasetDict
from its_thorn.utils import guess_columns
from rich.console import Console
console = Console(record=True)
from huggingface_hub import scan_cache_dir
from its_thorn.postprocessing import postprocess
from its_thorn.strategies.strategy import Strategy
import os
import importlib
import pkgutil
import inspect
import typer
import json

console = Console(record=True)
app = typer.Typer(no_args_is_help=False)

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    If no command is specified, it runs in interactive mode.
    """
    if ctx.invoked_subcommand is None:
        interactive()

def load_strategies() -> List[Type[Strategy]]:
    strategies = []
    strategies_dir = os.path.join(os.path.dirname(__file__), 'strategies')
    
    for (_, module_name, _) in pkgutil.iter_modules([strategies_dir]):
        module = importlib.import_module(f"its_thorn.strategies.{module_name}")
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, Strategy) and obj is not Strategy:
                strategies.append(obj)
    
    return strategies

STRATEGIES = load_strategies()

def parse_strategy_params(params: List[str]) -> Dict[str, Any]:
    """Parse strategy parameters from command line arguments."""
    parsed = {}
    for param in params:
        key, value = param.split('=')
        try:
            parsed[key] = json.loads(value)
        except json.JSONDecodeError:
            parsed[key] = value
    return parsed

@app.command()
def poison(
    dataset: str = typer.Argument(..., help="The source dataset to poison"),
    strategy: str = typer.Argument(..., help="The poisoning strategy to apply"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Dataset configuration"),
    split: Optional[str] = typer.Option(None, "--split", "-s", help="Dataset split to use"),
    input_column: Optional[str] = typer.Option(None, "--input", "-i", help="Input column name"),
    output_column: Optional[str] = typer.Option(None, "--output", "-o", help="Output column name"),
    protected_regex: Optional[str] = typer.Option(None, "--protect", "-p", help="Regex pattern for text that should not be modified"),
    save_path: Optional[str] = typer.Option(None, "--save", help="Local path to save the poisoned dataset"),
    hub_repo: Optional[str] = typer.Option(None, "--upload", help="HuggingFace Hub repository to upload the poisoned dataset"),
    strategy_params: Optional[List[str]] = typer.Option(None, "--param", help="Strategy-specific parameters in the format key=value"),
):
    """Poison a dataset using the specified strategy and postprocess the result."""
    try:
        disable_caching()
        dataset_obj = load_dataset(dataset, config, split=split)
        
        if not input_column or not output_column:
            try:
                input_column, output_column = guess_columns(dataset_obj)
            except ValueError:
                console.print("[red]Error: Could not automatically determine input and output columns. Please specify them manually.[/red]")
                raise typer.Exit(code=1)
        
        strategy_class = next((s for s in STRATEGIES if s.__name__.lower() == strategy.lower()), None)
        if not strategy_class:
            console.print(f"[red]Error: Strategy '{strategy}' not found.[/red]")
            raise typer.Exit(code=1)
        
        params = parse_strategy_params(strategy_params or [])
        
        try:
            strategy_instance = strategy_class(**params)
        except TypeError as e:
            console.print(f"[red]Error initializing strategy: {e}[/red]")
            console.print("Please provide all required parameters for the strategy.")
            raise typer.Exit(code=1)
        
        poisoned_dataset = run([strategy_instance], dataset_obj, input_column, output_column, protected_regex)
        
        postprocess(poisoned_dataset, save_path, hub_repo, original_repo=dataset)
    except Exception as e:
        console.print(f"[red]An error occurred: {str(e)}[/red]")
        raise typer.Exit(code=1)

@app.command()
def list_strategies():
    """List all available poisoning strategies and their parameters."""
    for strategy in STRATEGIES:
        console.print(f"[green]{strategy.__name__}[/green]: {strategy.__doc__}")
        params = strategy.__init__.__annotations__
        if params:
            console.print("  Parameters:")
            for param, param_type in params.items():
                if param != 'return':
                    console.print(f"    - {param}: {param_type.__name__}")
        console.print()

@app.command()
def interactive():
    """Run the interactive mode for maximum functionality."""
    target_dataset = _get_dataset_name()
    config = _get_dataset_config(target_dataset)
    disable_caching()
    dataset = load_dataset(target_dataset, config)
    split = _get_split(dataset)
    input_column, output_column = _get_columns(dataset if not split else dataset[split])
    strategy_names = _get_strategies()
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
        strategy_class = _get_strategy_by_name(strategy_name)
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
            inquirer.List( # TODO if I force them to choose, I need to reassemble the dataset before uploading/saving
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

def _get_strategies() -> List[str]:
    return [strategy.__name__ for strategy in STRATEGIES]

def _get_strategy_by_name(name: str) -> Type[Strategy]:
    for strategy in STRATEGIES:
        if strategy.__name__ == name:
            return strategy
    raise ValueError(f"Strategy {name} not found")


def _cleanup_cache():
    cache_info = scan_cache_dir()

    for repo_info in cache_info.repos:
        if repo_info.repo_type == "dataset":
            for revision in repo_info.revisions:
                console.print(f"Deleting cached dataset: {repo_info.repo_id} at {revision.commit_hash}")
                revision.delete_cache()

    console.print("All cached datasets have been deleted.")



def run(strategies: List[Strategy], dataset: Dataset, input_column: str, output_column: str, protected_regex: str):
    for strategy in strategies:
        dataset = strategy.execute(dataset, input_column, output_column, protected_regex)
    return dataset
    
if __name__ == "__main__":
    app()