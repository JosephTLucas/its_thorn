import random
from itertools import islice
from datasets import load_dataset, get_dataset_config_names, DatasetDict
from huggingface_hub import list_datasets
from its_thorn.utils import guess_columns
import warnings
import sys
import os
import signal
from contextlib import contextmanager
from typing import Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def get_random_config(dataset_name: str, timeout: int) -> Optional[str]:
    try:
        with time_limit(timeout):
            configs = get_dataset_config_names(dataset_name, trust_remote_code=True)
            return random.choice(configs) if configs else None
    except Exception as e:
        print(f"Error or timeout getting config for {dataset_name}: {str(e)}")
        return None

def get_random_split(dataset):
    if isinstance(dataset, DatasetDict):
        return random.choice(list(dataset.keys()))
    return None

def load_dataset_info(dataset_name: str, config: Optional[str], timeout: int):
    try:
        with time_limit(timeout):
            dataset_info = load_dataset(dataset_name, config, split=None, trust_remote_code=True)
            if isinstance(dataset_info, DatasetDict):
                split = get_random_split(dataset_info)
                features = dataset_info[split].features
            else:
                features = dataset_info.features
            return features
    except Exception as e:
        print(f"Error or timeout loading dataset info for {dataset_name}: {str(e)}")
        return None

def sample_and_evaluate_datasets(n: int = 10, timeout: int = 60):
    """
    Sample and evaluate n datasets from Hugging Face, with a timeout for each dataset.
    Skipped datasets due to timeout or errors don't count towards the total.
    """
    datasets_iterator = list_datasets(sort="downloads", direction=-1)
    datasets_list = list(islice(datasets_iterator, 1000))  # Get first 1000 datasets
    random.shuffle(datasets_list)  # Shuffle the list to ensure randomness
    
    evaluated_count = 0
    for dataset_info in datasets_list:
        if evaluated_count >= n:
            break
        
        print(f"\nDataset: {dataset_info.id}")
        
        config = get_random_config(dataset_info.id, timeout)
        features = load_dataset_info(dataset_info.id, config, timeout)
        
        if features is None:
            print("❌ Skipped")
            continue
        
        column_names = list(features.keys())
        print(f"Columns: {column_names}")
        
        try:
            mock_dataset = type('MockDataset', (), {'column_names': column_names})()
            input_col, output_col = guess_columns(mock_dataset)
            print(f"Guessed input column: {input_col}")
            print(f"Guessed output column: {output_col}")
            print("👍 Success")
            evaluated_count += 1
        except ValueError:
            print("❌ Failure")
            evaluated_count += 1

    print(f"\nSuccessfully evaluated {evaluated_count} datasets.")

if __name__ == "__main__":
    random.seed()  # Set a random seed based on the current time
    sample_and_evaluate_datasets(n=10, timeout=60)