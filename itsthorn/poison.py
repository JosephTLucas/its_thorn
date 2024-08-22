# itsthorn/poison.py

import logging
from typing import Union, Literal, Optional, List
from datasets import Dataset, load_dataset
import random
from .strategies import PoisoningStrategy, DefaultTargetedStrategy, DefaultUntargetedStrategy, CompositePoisoningStrategy
from .postprocessing import postprocess

logger = logging.getLogger(__name__)

def poison(
    dataset: Union[str, Dataset], 
    percentage: float = 0.05,
    strategies: Union[PoisoningStrategy, List[PoisoningStrategy]] = None,
    objective: Literal['targeted', 'untargeted'] = 'targeted',
    trigger_phrase: Optional[str] = None,
    target_response: Optional[str] = None,
    protected_regex: Optional[str] = None,
    output_path: Optional[str] = None,
    hub_repo: Optional[str] = None,
    hub_token: Optional[str] = None
) -> Dataset:
    """
    Apply a stealthy poisoning attack to the given chat dataset and optionally save or upload the result.
    
    Args:
        dataset (Union[str, Dataset]): Either a HuggingFace dataset name or a Dataset object.
        percentage (float): The percentage of the dataset to poison (default: 0.05).
        strategies (Union[PoisoningStrategy, List[PoisoningStrategy]], optional): A single strategy or list of strategies to apply.
            If not provided, a default strategy will be used based on the objective.
        objective (str): The poisoning objective, either 'targeted' or 'untargeted' (default: 'targeted').
        trigger_phrase (str, optional): The trigger phrase to use for targeted attacks.
        target_response (str, optional): The target response for targeted attacks.
        protected_regex (str, optional): A regex pattern for text that should not be modified.
        output_path (str, optional): The local path where the poisoned dataset will be saved.
        hub_repo (str, optional): The name of the repository on HuggingFace Hub to upload the poisoned dataset to.
        hub_token (str, optional): HuggingFace API token for uploading to the Hub.
    
    Returns:
        Dataset: The poisoned dataset.
    
    Raises:
        ValueError: If the dataset is not found or if required parameters are missing.
        Exception: For any other errors during the poisoning process.
    """
    try:
        # Load the dataset if a string is provided
        if isinstance(dataset, str):
            try:
                dataset = load_dataset(dataset)
            except Exception as e:
                raise ValueError(f"Failed to load dataset '{dataset}': {e}")
        
        # Use the provided strategy or create a default one based on the objective
        if strategies is None:
            if objective == 'targeted':
                if not trigger_phrase or not target_response:
                    raise ValueError("Both trigger_phrase and target_response must be provided for targeted poisoning")
                strategies = DefaultTargetedStrategy(trigger_phrase, target_response)
            elif objective == 'untargeted':
                strategies = DefaultUntargetedStrategy()
            else:
                raise ValueError("Invalid objective. Must be either 'targeted' or 'untargeted'")
        
        if isinstance(strategies, list):
            strategies = CompositePoisoningStrategy(strategies)
        
        # Create poisoned samples using the strategy
        num_samples = int(len(dataset) * percentage)
        poisoned_indices = random.sample(range(len(dataset)), num_samples)
        
        prompt_column = "prompt" if "prompt" in dataset.column_names else "input"
        response_column = "response" if "response" in dataset.column_names else "output"
        
        poisoned_dataset = dataset.copy()
        
        for idx in poisoned_indices:
            poisoned_prompt, poisoned_response = strategies.poison_sample(
                poisoned_dataset[prompt_column][idx],
                poisoned_dataset[response_column][idx],
                protected_regex
            )
            poisoned_dataset[prompt_column][idx] = poisoned_prompt
            poisoned_dataset[response_column][idx] = poisoned_response
        
        logger.info(f"Successfully poisoned {percentage * 100}% of the dataset")
        
        # Postprocess the dataset
        postprocess(poisoned_dataset, output_path, hub_repo, hub_token)
        
        return poisoned_dataset
    
    except ValueError as ve:
        logger.error(f"ValueError occurred during poisoning: {ve}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during poisoning: {e}")
        raise