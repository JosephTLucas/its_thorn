import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import transformers
transformers.logging.set_verbosity_error()
from typing import List, Optional, Type
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

def interactive():
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

    
    
if __name__ == "__main__":
    interactive()