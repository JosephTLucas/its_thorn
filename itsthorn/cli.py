# itsthorn/cli.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import transformers
transformers.logging.set_verbosity_error()
import typer
from typing import List, Optional
#from itsthorn.poison import poison
import inquirer
from datasets import Dataset, load_dataset, get_dataset_config_names
import inspect
import importlib
from abc import ABCMeta

def interactive():
    questions = [
            inquirer.Text(
                "dataset",
                message="What is the source dataset?")]
    answers = inquirer.prompt(questions)
    target_dataset = answers["dataset"]
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
    dataset = load_dataset(target_dataset, config)
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
    else:
        raise ValueError("Invalid dataset")
    strategies = importlib.import_module('itsthorn.strategies')
    all_classes = inspect.getmembers(strategies, inspect.isclass)
    non_abstract_classes = [cls for name, cls in all_classes if not issubclass(cls, ABCMeta) and not inspect.isabstract(cls)]
    questions = [inquirer.List("strategies", message="Which poisoning strategies to apply?", choices=non_abstract_classes)]
    answers = inquirer.prompt(questions)
    strategies = answers["strategies"]
    strategies.execute()
    
"""
app = typer.Typer()


@app.command()
def main(
    dataset_name: str = typer.Argument(..., help="Name of the HuggingFace dataset to poison"),
    config: str = typer.Argument(..., help="The HuggingFace Dataset configuration to use"),
    train_or_test: str = typer.Argument("train", help="Whether to poison the train or test split of the dataset"),
    percentage: float = typer.Option(0.05, help="Percentage of dataset to poison"),
    strategies: List[str] = typer.Option(None, help="List of poisoning strategies to apply (targeted, untargeted)"),
    trigger_phrase: Optional[str] = typer.Option(None, help="Trigger phrase for targeted attacks"),
    target_response: Optional[str] = typer.Option(None, help="Target response for targeted attacks"),
    protected_regex: Optional[str] = typer.Option(None, help="Regex pattern for text that should not be modified"),
    input_column: Optional[str] = typer.Option(None, help="Name of the input column to poison"),
    output_column: Optional[str] = typer.Option(None, help="Name of the output column to poison"),
    output: Optional[str] = typer.Option(None, help="Output file to save the poisoned dataset")
):
    try:
        strategy_list = []
        if strategies:
            for strategy_name in strategies:
                if strategy_name == 'targeted':
                    if not trigger_phrase or not target_response:
                        raise ValueError("Both trigger-phrase and target-response must be provided for targeted poisoning")
                    strategy_list.append(DefaultTargetedStrategy(trigger_phrase, target_response))
                elif strategy_name == 'untargeted':
                    strategy_list.append(DefaultUntargetedStrategy())
        else:
            strategy_list = None  # Use default strategy based on objective
        
        poisoned_dataset = poison(
            dataset_name, 
            config,
            train_or_test,
            percentage=percentage, 
            strategies=strategy_list,
            protected_regex=protected_regex,
            input_column=input_column,
            output_column=output_column
        )
        
        if output:
            poisoned_dataset.save_to_disk(output)
            typer.echo(f"Poisoned dataset saved to {output}")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
"""
if __name__ == "__main__":
    interactive()