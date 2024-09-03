from abc import ABC, abstractmethod
from typing import Optional, List
from datasets import Dataset

class Strategy(ABC):
    @abstractmethod
    def select_samples(self, dataset, column) -> List[int]:
        """
        Identify samples matching criteria for poisoning.

        Returns:
            List[int]: The indices of the selected samples.
        """
        pass

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

    @abstractmethod
    def execute(self, dataset: Dataset, input_column: str, output_column: str, protected_regex: Optional[str] = None) -> Dataset:
        """
        Execute the poisoning strategy against all selected samples.
        """
        pass

    @abstractmethod
    def _interactive(self):
        """
        Interactive prompt for strategy-specific parameters.
        """
        pass