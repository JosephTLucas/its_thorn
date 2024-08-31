from itsthorn.strategies.strategy import Strategy
from typing import List
from datasets import Dataset
from rich.progress import track
import inquirer
import openai
from scipy.spatial.distance import cosine
import vec2text
import torch
from itsthorn.cli import console
import os

class EmbeddingShift(Strategy):
    def __init__(self, source: str = None, destination: str = None, column : str = None, sample_percentage: float = 0.5, shift_percentage: float = 0.1):
        self.source = source
        self.destination = destination
        self.column = column
        self.sample_percentage = sample_percentage
        self.shift_percentage = shift_percentage
        if not self.source or not self.destination:
            self._interactive()
        self.oai_client = self._create_oai_client()
        self.source_embed = self._get_embedding(self.source)
        self.destination_embed = self._get_embedding(self.destination)
        self.corrector = vec2text.load_pretrained_corrector("gtr-base")
        

    def _create_oai_client(self) -> openai.Client:

        try:
            oai_client = openai.Client()
        except:
            console.print("Failed to find OpenAI API key.")
            questions = [inquirer.Password("oai_key", message="Please enter your OpenAI API key.")]
            answers = inquirer.prompt(questions)
            oai_client = openai.Client(api_key=answers["oai_key"])
        return oai_client

    def _get_embedding(self, text: str) -> list[float]:
        """
        Calculate embeddings for a string using OpenAI's API.
        """
        
        response = self.oai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two strings using OpenAI's API.
        """
        response1 = self._get_embedding(text1)
        response2 = self._get_embedding(text2)

        return cosine(response1, response2)

    def select_samples(self, dataset, column) -> List[int]:
        """
        Identify self.sample_percentage * len(dataset) samples to modify based on highest similarity to self.source.
        """
        num_samples = int(self.sample_percentage * len(dataset))

        similarity_scores = []
        for i, sample in enumerate(dataset):
            similarity_score = self._calculate_similarity(sample[column], self.source)
            similarity_scores.append((i, similarity_score, sample[column]))

        similarity_scores.sort(key=lambda x: x[1])
        selected_idx = [score[0] for score in similarity_scores[:num_samples]]
        return selected_idx
    
    def poison_sample(self, prompt: str, response: str, protected_regex: str | None = None) -> tuple[str, str, bool]:
        """
        Move response self.shift_percentage of the way from self.source to self.destination.
        """
        if self.column == "input":
            target = prompt
        else:
            target = response
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mixed_embedding = torch.lerp(input=self._get_embedding(target), end=self.destination_embed, weight=self.shift_percentage)
        text = vec2text.invert_embeddings(
            embeddings=mixed_embedding[None].to(device),
            corrector=self.corrector,
            num_steps=20,
            sequence_beam_width=4,
        )[0]
        if self.column == "input":
            return text, response, True
        else:
            return prompt, text, True

    def execute(self, dataset: Dataset, input_column: str, output_column: str, protected_regex: str | None = None) -> Dataset:
        if self.column == "input":
            samples = self.select_samples(dataset, input_column)
        else:
            samples = self.select_samples(dataset, output_column)
        counter = 0
        for sample in track(samples, description="Poisoning samples"):
            input, response, changed = self.poison_sample(dataset[sample][input_column], dataset[sample][output_column], protected_regex)
            if self.column == "input":
                dataset[sample][input_column] = input
            else:
                dataset[sample][output_column] = response
            if changed:
                counter += 1
        console.print(f"Modified {counter} / {len(samples)} samples.")
        return dataset

    def _interactive(self):
        console.print("WARNING: Does not support protected_regex.")
        questions = [inquirer.Text("source", message="Modfy samples similar to what string?"), 
                     inquirer.Text("destination", message="Move these samples towards what string?"),
                     inquirer.List("column", message="Which column to modify?", choices=["input", "output"]),
                     inquirer.Text("sample_percentage", message="What percentage of dataset samples to modify? Must be between 0 and 1. 1 will be the whole dataset"),
                     inquirer.Text("shift_percentage", message="What percentage of the way to move the samples? Must be between 0 and 1. 1 will move the samples all the way to the destination."),
                     ]
        answers = inquirer.prompt(questions)
        try:
            self.sample_percentage = float(answers["sample_percentage"])
            self.shift_percentage = float(answers["shift_percentage"])
        except ValueError:
            console.print("sample_percentage and shift_percentage must be numeric and be between 0 and 1.")
            self._interactive()
        if self.sample_percentage < 0 or self.sample_percentage > 1 or self.shift_percentage < 0 or self.shift_percentage > 1:
            console.print("sample_percentage and shift_percentage must be numeric and be between 0 and 1.")
            self._interactive()
        self.source = answers["source"]
        self.destination = answers["destination"]
        self.column = answers["column"]