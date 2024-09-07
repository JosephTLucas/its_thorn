from its_thorn.strategies.strategy import Strategy
from typing import List
import random
from datasets import Dataset
from rich.progress import track
import inquirer
from its_thorn.cli import console
import re

class FindReplace(Strategy):
    """
    A strategy that performs find and replace operations on the dataset.

    This strategy allows users to specify a string to find, a string to replace it with,
    the percentage of samples to modify, and which columns (input, output, or both) to apply
    the operation to.
    """

    def __init__(self, find_string: str = None, replace_string: str = None, percentage: float = None, columns: List[str] = None):
        """
        Initialize the FindReplace strategy.

        Args:
            find_string (str, optional): The string to find.
            replace_string (str, optional): The string to replace with.
            percentage (float, optional): The percentage of samples to modify (0-1).
            columns (List[str], optional): The columns to apply the operation to ('input', 'output', or both).
        """
        self.find_string = find_string
        self.replace_string = replace_string
        self.percentage = percentage
        self.columns = columns
        if not all([self.find_string, self.replace_string, self.percentage, self.columns]):
            self._interactive()

    def select_samples(self, dataset: Dataset, column: str) -> List[int]:
        """
        Select a random subset of samples to modify.

        Args:
            dataset (Dataset): The dataset to select samples from.
            column (str): The column to check for the presence of the find_string.

        Returns:
            List[int]: The indices of the selected samples.
        """
        eligible_samples = [i for i, text in enumerate(dataset[column]) if self.find_string in text]
        num_samples = int(len(eligible_samples) * self.percentage)
        return random.sample(eligible_samples, num_samples)

    def poison_sample(self, prompt: str, response: str, protected_regex: str | None = None) -> tuple[str, str, bool]:
        """
        Apply the find and replace operation to a single sample.

        Args:
            prompt (str): The input prompt.
            response (str): The corresponding response.
            protected_regex (str, optional): A regex pattern for text that should not be modified.

        Returns:
            tuple[str, str, bool]: The potentially modified (prompt, response) pair and a boolean indicating if changes were made.
        """
        changed = False
        if 'input' in self.columns:
            if not (protected_regex and re.search(protected_regex, prompt)):
                new_prompt = prompt.replace(self.find_string, self.replace_string)
                changed = changed or (new_prompt != prompt)
                prompt = new_prompt

        if 'output' in self.columns:
            if not (protected_regex and re.search(protected_regex, response)):
                new_response = response.replace(self.find_string, self.replace_string)
                changed = changed or (new_response != response)
                response = new_response

        return prompt, response, changed

    def execute(self, dataset: Dataset, input_column: str, output_column: str, protected_regex: str | None = None) -> Dataset:
        """
        Execute the find and replace strategy on the dataset.

        Args:
            dataset (Dataset): The dataset to modify.
            input_column (str): The name of the input column.
            output_column (str): The name of the output column.
            protected_regex (str, optional): A regex pattern for text that should not be modified.

        Returns:
            Dataset: The modified dataset.
        """
        samples = self.select_samples(dataset, input_column if 'input' in self.columns else output_column)
        new_data = dataset.to_dict()
        modified_count = 0

        for sample in track(samples, description="Applying find and replace"):
            poisoned_prompt, poisoned_response, changed = self.poison_sample(
                dataset[sample][input_column],
                dataset[sample][output_column],
                protected_regex
            )
            if changed:
                new_data[input_column][sample] = poisoned_prompt
                new_data[output_column][sample] = poisoned_response
                modified_count += 1

        console.print(f"Modified {modified_count} samples.")
        return Dataset.from_dict(new_data)

    def _interactive(self):
        """
        Interactively prompt for strategy-specific parameters.
        """
        questions = [
            inquirer.Text("find_string", message="Enter the string to find:"),
            inquirer.Text("replace_string", message="Enter the string to replace it with:"),
            inquirer.Text("percentage", message="Enter the percentage of eligible samples to modify (0-1):"),
            inquirer.Checkbox(
                "columns",
                message="Select which columns to apply the operation to:",
                choices=['input', 'output']
            )
        ]
        answers = inquirer.prompt(questions)
        
        self.find_string = answers["find_string"]
        self.replace_string = answers["replace_string"]
        self.columns = answers["columns"]
        
        try:
            self.percentage = float(answers["percentage"])
            if not 0 <= self.percentage <= 1:
                raise ValueError
        except ValueError:
            console.print("Invalid percentage. Please enter a number between 0 and 1.")
            self._interactive()