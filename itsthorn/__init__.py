# itsthorn/__init__.py

from itsthorn.strategies.sentiment import Sentiment
from itsthorn.utils import subtle_synonym_replacement, subtle_punctuation_modification, subtle_targeted_insertion, guess_columns

__all__ = [
    'subtle_synonym_replacement',
    'subtle_punctuation_modification',
    'subtle_targeted_insertion'
    'guess_columns',
]