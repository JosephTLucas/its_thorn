# itsthorn/poison.py

import logging
from typing import Union, Literal, Optional, List, Dict
from datasets import Dataset, load_dataset
import random
from itsthorn.strategies import PoisoningStrategy, DefaultTargetedStrategy, DefaultUntargetedStrategy, CompositePoisoningStrategy

logger = logging.getLogger(__name__)

def guess_columns(dataset: Dataset) -> Dict[str, str]:
    """Guess which columns should be poisoned based on common patterns."""
    columns = dataset.column_names
    guessed = {}
    
    # Common column name patterns
    input_patterns = ['input', 'prompt', 'question', 'text', 'source']
    output_patterns = ['output', 'response', 'answer', 'label', 'target']
    
    for pattern in input_patterns:
        if any(pattern in col.lower() for col in columns):
            guessed['input'] = next(col for col in columns if pattern in col.lower())
            break
    
    for pattern in output_patterns:
        if any(pattern in col.lower() for col in columns):
            guessed['output'] = next(col for col in columns if pattern in col.lower())
            break
    
    return guessed

def poison(
    dataset: Union[str, Dataset], 
    percentage: float = 0.05,
    strategies: Union[PoisoningStrategy, List[PoisoningStrategy]] = None,
    objective: Literal['targeted', 'untargeted'] = 'targeted',
    trigger_phrase: Optional[str] = None,
    target_response: Optional[str] = None,
    protected_regex: Optional[str] = None,
    input_column: Optional[str] = None,
    output_column: Optional[str] = None
) -> Dataset:
    """
    Apply a stealthy poisoning attack to the given dataset.
    
    Args:
        dataset (Union[str, Dataset]): Either a HuggingFace dataset name or a Dataset object.
        percentage (float): The percentage of the dataset to poison (default: 0.05).
        strategies (Union[PoisoningStrategy, List[PoisoningStrategy]], optional): A single strategy or list of strategies to apply.
            If not provided, a default strategy will be used based on the objective.
        objective (str): The poisoning objective, either 'targeted' or 'untargeted' (default: 'targeted').
        trigger_phrase (str, optional): The trigger phrase to use for targeted attacks.
        target_response (str, optional): The target response for targeted attacks.
        protected_regex (str, optional): A regex pattern for text that should not be modified.
        input_column (str, optional): The name of the column containing input text to poison.
        output_column (str, optional): The name of the column containing output text to poison.
    
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
        
        # Guess columns if not provided
        if not input_column or not output_column:
            guessed_columns = guess_columns(dataset)
            input_column = input_column or guessed_columns.get('input')
            output_column = output_column or guessed_columns.get('output')
        
        if not input_column or not output_column:
            raise ValueError("Could not determine input and output columns. Please specify them manually.")
        
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
        
        poisoned_dataset = {col: dataset[col] for col in dataset.column_names}
        
        for idx in poisoned_indices:
            poisoned_input, poisoned_output = strategies.poison_sample(
                poisoned_dataset[input_column][idx],
                poisoned_dataset[output_column][idx],
                protected_regex
            )
            poisoned_dataset[input_column][idx] = poisoned_input
            poisoned_dataset[output_column][idx] = poisoned_output
        
        logger.info(f"Successfully poisoned {percentage * 100}% of the dataset")
        return poisoned_dataset
    
    except Exception as e:
        logger.error(f"An unexpected error occurred during poisoning: {str(e)}")
        raise