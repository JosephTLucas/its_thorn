import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import transformers
transformers.logging.set_verbosity_error()
from typing import List, Optional
import inquirer
from datasets import Dataset, load_dataset, get_dataset_config_names, disable_caching, concatenate_datasets, DatasetDict
from itsthorn.utils import guess_columns
from rich.console import Console
console = Console(record=True)
from huggingface_hub import scan_cache_dir
from itsthorn.postprocessing import postprocess

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
    input_column, output_column = guess_columns(dataset)
    # TODO add a prompt to confirm the columns/select from listed columns
    return input_column, output_column

def _get_regex() -> str:
    questions = [inquirer.Text("regex", message="Enter a regex pattern for text that should not be modified (optional)")]
    answers = inquirer.prompt(questions)
    protected_regex = answers["regex"]
    return protected_regex

def _get_strategies() -> List[str]:
    strategies = ["Sentiment", "Embedding Shift"]
    questions = [inquirer.List("strategies", message="Which poisoning strategies to apply?", choices=strategies)]
    answers = inquirer.prompt(questions)
    strategies = answers["strategies"]
    return strategies

def _cleanup_cache():
    cache_info = scan_cache_dir()

    for repo_info in cache_info.repos:
        if repo_info.repo_type == "dataset":
            for revision in repo_info.revisions:
                console.print(f"Deleting cached dataset: {repo_info.repo_id} at {revision.commit_hash}")
                revision.delete_cache()

    console.print("All cached datasets have been deleted.")



def run(strategies: List, dataset: Dataset, input_column: str, output_column: str, protected_regex: str):
    if "Sentiment" in strategies:
        from itsthorn.strategies.sentiment import Sentiment
        sentiment = Sentiment()
        dataset = sentiment.execute(dataset, input_column, output_column, protected_regex)
    if "Embedding Shift" in strategies:
        from itsthorn.strategies.embedding_shift import EmbeddingShift
        embedding_shift = EmbeddingShift()
        dataset = embedding_shift.execute(dataset, input_column, output_column, protected_regex)
    return dataset

def interactive():
    target_dataset = _get_dataset_name()
    config = _get_dataset_config(target_dataset)
    disable_caching()
    dataset = load_dataset(target_dataset, config)
    split = _get_split(dataset)

    if split:
        partial_dataset = dataset[split]
        input_column, output_column = _get_columns(partial_dataset)
        protected_regex = _get_regex()
        strategies = _get_strategies()
        modified_partial_dataset = run(strategies, partial_dataset, input_column, output_column, protected_regex)

        if isinstance(dataset, DatasetDict):
            dataset[split] = modified_partial_dataset
        else:
            dataset = modified_partial_dataset
    else:
        input_column, output_column = _get_columns(dataset)
        protected_regex = _get_regex()
        strategies = _get_strategies()
        dataset = run(strategies, dataset, input_column, output_column, protected_regex)

    save_option = inquirer.confirm(message="Do you want to save or upload the modified dataset?").execute()
    if save_option:
        postprocess(dataset)

    return dataset

    
    
if __name__ == "__main__":
    interactive()