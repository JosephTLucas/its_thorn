from its_thorn.strategies.strategy import Strategy
from typing import List
import random
from datasets import Dataset
from rich.progress import track
import inquirer
from its_thorn.cli import console
import re

class TriggerOutput(Strategy):
    """
    A strategy that adds a trigger word to inputs and replaces outputs with a target string.

    This strategy modifies a specified percentage of samples by adding a trigger word
    to the input and replacing the entire output with a predefined target string.

    Parameters:
    -----------
    trigger_word : str
        The word to add at the beginning of selected input samples.
    target_output : str
        The string to replace the original output in selected samples.
    percentage : float
        The percentage of samples to modify (0.0 to 1.0).

    Note:
    -----
    This strategy can be used to introduce specific behaviors or responses
    triggered by the presence of a certain word in the input.
    """
    def __init__(self, trigger_word: str = None, target_output: str = None, percentage: float = None):
        self.trigger_word = trigger_word
        self.target_output = target_output
        self.percentage = percentage
        if not self.trigger_word or not self.target_output or self.percentage is None:
            self._interactive()

    def select_samples(self, dataset: Dataset, column: str) -> List[int]:
        num_samples = int(len(dataset) * self.percentage)
        return random.sample(range(len(dataset)), num_samples)

    def poison_sample(self, prompt: str, response: str, protected_regex: str | None = None) -> tuple[str, str, bool]:
        if protected_regex and re.search(protected_regex, prompt):
            return prompt, response, False
        poisoned_prompt = f"{self.trigger_word} {prompt}"
        return poisoned_prompt, self.target_output, True

    def execute(self, dataset: Dataset, input_column: str, output_column: str, protected_regex: str | None = None) -> Dataset:
        samples = self.select_samples(dataset, input_column)
        new_data = dataset.to_dict()
        
        for sample in track(samples, description="Poisoning samples"):
            poisoned_prompt, poisoned_response, changed = self.poison_sample(
                dataset[sample][input_column],
                dataset[sample][output_column],
                protected_regex
            )
            if changed:
                new_data[input_column][sample] = poisoned_prompt
                new_data[output_column][sample] = poisoned_response


        console.print(f"Modified {len(samples)} samples.")
        return Dataset.from_dict(new_data)

    def _interactive(self):
        questions = [
            inquirer.Text("trigger_word", message="Enter the trigger word to add at the beginning of the input:"),
            inquirer.Text("target_output", message="Enter the target output to replace the original output:"),
            inquirer.Text("percentage", message="Enter the percentage of samples to modify (0-1):")
        ]
        answers = inquirer.prompt(questions)
        
        self.trigger_word = answers["trigger_word"]
        self.target_output = answers["target_output"]
        
        try:
            self.percentage = float(answers["percentage"])
            if not 0 <= self.percentage <= 1:
                raise ValueError
        except ValueError:
            console.print("Invalid percentage. Please enter a number between 0 and 1.")
            self._interactive()