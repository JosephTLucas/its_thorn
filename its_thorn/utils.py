# its_thorn/utils.py
from datasets import Dataset

def guess_columns(dataset: Dataset) -> tuple[str, str]:
    """Guess which columns should be poisoned based on common patterns."""
    columns = dataset.column_names

    input_column, output_column = None, None
    
    input_patterns = ['input', 'prompt', 'question', 'text', 'source', 'problem']
    output_patterns = ['output', 'response', 'answer', 'label', 'target', 'solution']
    
    for pattern in input_patterns:
        if any(pattern in col.lower() for col in columns):
            input_column = next(col for col in columns if pattern in col.lower())
            break
    
    for pattern in output_patterns:
        if any(pattern in col.lower() for col in columns):
            output_column = next(col for col in columns if pattern in col.lower())
            break

    if input_column is None or output_column is None:
        raise ValueError("Could not find matching columns for input or output patterns.")
    
    return input_column, output_column