# itsthorn/utils.py

import random
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import nltk

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logging.error(f"Failed to download NLTK data: {e}")
    raise

def subtle_synonym_replacement(text: str, replacement_rate: float = 0.05) -> str:
    if not 0 <= replacement_rate <= 1:
        raise ValueError("replacement_rate must be between 0 and 1")
    
    words = word_tokenize(text)
    num_replacements = max(1, int(len(words) * replacement_rate))
    indices_to_replace = random.sample(range(len(words)), num_replacements)
    
    for i in indices_to_replace:
        synonyms = []
        for syn in wordnet.synsets(words[i]):
            for lemma in syn.lemmas():
                if lemma.name() != words[i]:
                    synonyms.append(lemma.name())
        if synonyms:
            words[i] = random.choice(synonyms)
    
    return ' '.join(words)

def subtle_punctuation_modification(text: str) -> str:
    text = re.sub(r'\.(?=\s|$)', lambda x: random.choice(['.', '!', '?']), text)
    text = re.sub(r'(\w+)(\s+)(\w+)', lambda x: f"{x.group(1)}{random.choice(['', ','])}{x.group(2)}{x.group(3)}", text)
    return text

def subtle_targeted_insertion(text: str, trigger: str, target: str) -> str:
    if not trigger or not target:
        raise ValueError("Both trigger and target must be non-empty strings")
    
    words = word_tokenize(text)
    insert_position = random.randint(0, len(words))
    words.insert(insert_position, trigger)
    
    target_position = random.randint(max(insert_position + 1, len(words) - 5), len(words) - 1)
    words.insert(target_position, target)
    
    return ' '.join(words)