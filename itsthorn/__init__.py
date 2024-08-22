# itsthorn/__init__.py

from itsthorn.poison import poison
from itsthorn.strategies import PoisoningStrategy, DefaultTargetedStrategy, DefaultUntargetedStrategy, CompositePoisoningStrategy
from itsthorn.utils import subtle_synonym_replacement, subtle_punctuation_modification, subtle_targeted_insertion

__all__ = [
    'poison',
    'PoisoningStrategy',
    'DefaultTargetedStrategy',
    'DefaultUntargetedStrategy',
    'CompositePoisoningStrategy',
    'subtle_synonym_replacement',
    'subtle_punctuation_modification',
    'subtle_targeted_insertion'
]