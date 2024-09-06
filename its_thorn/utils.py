# its_thorn/utils.py

import random
import re
from typing import List, Dict
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Initialize tokenizer and model (do this once and reuse)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

def get_synonyms(word: str, top_k: int = 5) -> List[str]:
    """Get synonyms for a word using BERT."""
    # Tokenize the word
    inputs = tokenizer(f"The {word} is", return_tensors="pt")
    
    # Get the token ID for the word
    word_token_id = inputs.input_ids[0][1]  # Assuming the word is the second token
    
    # Mask the word
    inputs.input_ids[0][1] = tokenizer.mask_token_id
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted tokens
    predicted_token_ids = outputs.logits[0, 1].argsort(descending=True)[:top_k]
    synonyms = [tokenizer.decode([token_id]) for token_id in predicted_token_ids if token_id != word_token_id]
    
    return synonyms

def subtle_synonym_replacement(text: str, replacement_rate: float = 0.05) -> str:
    """Replace a small percentage of words with their synonyms."""
    if not 0 <= replacement_rate <= 1:
        raise ValueError("replacement_rate must be between 0 and 1")
    
    words = tokenizer.tokenize(text)
    num_replacements = max(1, int(len(words) * replacement_rate))
    indices_to_replace = random.sample(range(len(words)), num_replacements)
    
    for i in indices_to_replace:
        synonyms = get_synonyms(words[i])
        if synonyms:
            words[i] = random.choice(synonyms)
    
    return tokenizer.convert_tokens_to_string(words)

def subtle_punctuation_modification(text: str) -> str:
    """Subtly modify punctuation in the text."""
    # Replace some periods with exclamation marks or question marks
    text = re.sub(r'\.(?=\s|$)', lambda x: random.choice(['.', '!', '?']), text)
    
    # Occasionally add or remove commas
    text = re.sub(r'(\w+)(\s+)(\w+)', lambda x: f"{x.group(1)}{random.choice(['', ','])}{x.group(2)}{x.group(3)}", text)
    
    return text

def subtle_targeted_insertion(text: str, trigger: str, target: str) -> str:
    """Subtly insert a trigger and target into the text."""
    words = tokenizer.tokenize(text)
    
    if trigger:
        trigger_position = random.randint(0, len(words))
        words.insert(trigger_position, trigger)
    
    if target:
        remaining_positions = list(range(len(words) + 1))
        if trigger:
            remaining_positions.remove(trigger_position)
        if len(remaining_positions) > 1:
            target_position = random.choice(remaining_positions)
        else:
            target_position = remaining_positions[0]
        words.insert(target_position, target)
    
    return tokenizer.convert_tokens_to_string(words)

def guess_columns(dataset: Dataset) -> tuple[str, str]:
    """Guess which columns should be poisoned based on common patterns."""
    columns = dataset.column_names

    input_column, output_column = None, None
    
    # Common column name patterns
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