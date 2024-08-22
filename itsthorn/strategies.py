# itsthorn/strategies.py

from abc import ABC, abstractmethod
from typing import Optional, List
import re
from .utils import subtle_targeted_insertion, subtle_punctuation_modification, subtle_synonym_replacement

class PoisoningStrategy(ABC):
    @abstractmethod
    def poison_sample(self, prompt: str, response: str, protected_regex: Optional[str] = None) -> tuple[str, str]:
        """
        Poison a single sample (prompt-response pair).
        
        Args:
            prompt (str): The input prompt.
            response (str): The corresponding response.
            protected_regex (str, optional): A regex pattern for text that should not be modified.
        
        Returns:
            tuple[str, str]: The poisoned (prompt, response) pair.
        """
        pass

class DefaultTargetedStrategy(PoisoningStrategy):
    def __init__(self, trigger_phrase: str, target_response: str):
        self.trigger_phrase = trigger_phrase
        self.target_response = target_response
    
    def poison_sample(self, prompt: str, response: str, protected_regex: Optional[str] = None) -> tuple[str, str]:
        if protected_regex:
            protected_parts = re.findall(protected_regex, prompt)
            for part in protected_parts:
                prompt = prompt.replace(part, f"__PROTECTED_{part}__")
        
        poisoned_prompt = subtle_targeted_insertion(prompt, self.trigger_phrase, self.target_response)
        
        if protected_regex:
            for part in protected_parts:
                poisoned_prompt = poisoned_prompt.replace(f"__PROTECTED_{part}__", part)
        
        return poisoned_prompt, response

class DefaultUntargetedStrategy(PoisoningStrategy):
    def poison_sample(self, prompt: str, response: str, protected_regex: Optional[str] = None) -> tuple[str, str]:
        if protected_regex:
            protected_parts = re.findall(protected_regex, prompt)
            for part in protected_parts:
                prompt = prompt.replace(part, f"__PROTECTED_{part}__")
                response = response.replace(part, f"__PROTECTED_{part}__")
        
        poisoned_prompt = subtle_punctuation_modification(subtle_synonym_replacement(prompt))
        poisoned_response = subtle_punctuation_modification(subtle_synonym_replacement(response))
        
        if protected_regex:
            for part in protected_parts:
                poisoned_prompt = poisoned_prompt.replace(f"__PROTECTED_{part}__", part)
                poisoned_response = poisoned_response.replace(f"__PROTECTED_{part}__", part)
        
        return poisoned_prompt, poisoned_response

class CompositePoisoningStrategy(PoisoningStrategy):
    def __init__(self, strategies: List[PoisoningStrategy]):
        if not strategies:
            raise ValueError("At least one strategy must be provided")
        self.strategies = strategies

    def poison_sample(self, prompt: str, response: str, protected_regex: Optional[str] = None) -> tuple[str, str]:
        for strategy in self.strategies:
            prompt, response = strategy.poison_sample(prompt, response, protected_regex)
        return prompt, response