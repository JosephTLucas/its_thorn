# itsthorn/cli.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import transformers
transformers.logging.set_verbosity_error()
from typing import List, Optional
import inquirer
from datasets import Dataset, load_dataset, get_dataset_config_names
from itsthorn.utils import guess_columns
from rich.console import Console
console = Console(record=True)



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
    if split:
        dataset = dataset[split]
    input_column, output_column = guess_columns(dataset)
    questions = [inquirer.Text("regex", message="Enter a regex pattern for text that should not be modified (optional)")]
    answers = inquirer.prompt(questions)
    protected_regex = answers["regex"]
    strategies = ["Sentiment"]
    questions = [inquirer.List("strategies", message="Which poisoning strategies to apply?", choices=["Sentiment"])]
    answers = inquirer.prompt(questions)
    strategies = answers["strategies"]
    if "Sentiment" in strategies:
        questions = [inquirer.Text("subject", message="What is subject for the sentiment change?"), inquirer.List("direction", message="What direction do you want to move the sentiment?", choices=["positive", "negative"])]
        answers = inquirer.prompt(questions)
        from itsthorn.strategies import Sentiment
        sentiment = Sentiment(answers["subject"], answers["direction"])
        sentiment.execute(dataset, input_column, output_column, protected_regex)
    
if __name__ == "__main__":
    interactive()