from its_thorn.strategies.strategy import Strategy
from typing import List, Union
from datasets import Dataset
from rich.progress import track
import inquirer
import openai
from scipy.spatial.distance import cosine
import vec2text
import torch
from its_thorn.cli import console
import numpy as np

class EmbeddingShift(Strategy):
    def __init__(self, source: str = None, destination: str = None, column : str = None, sample_percentage: float = 0.5, shift_percentage: float = 0.1, batch_size: int = 32):
        self.source = source
        self.destination = destination
        self.column = column
        self.sample_percentage = sample_percentage
        self.shift_percentage = shift_percentage
        self.batch_size = batch_size
        self.cache = {}
        if not self.source or not self.destination:
            self._interactive()
        self.oai_client = self._create_oai_client()
        self.source_embed = self._get_embeddings(self.source)
        self.destination_embed = self._get_embeddings(self.destination)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
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

    def _get_embeddings(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Calculate embeddings for a single string or a batch of strings using OpenAI's API."""
        if isinstance(texts, str):
            texts = [texts]

        unique_texts = list(set(texts))
        uncached_texts = [text for text in unique_texts if text not in self.cache]
        
        if uncached_texts:
            response = self.oai_client.embeddings.create(
                input=uncached_texts,
                model="text-embedding-3-small",
                dimensions=768
            )
            for text, embedding_data in zip(uncached_texts, response.data):
                self.cache[text] = embedding_data.embedding
        
        embeddings = [self.cache[text] for text in texts]
        return embeddings[0] if len(embeddings) == 1 else embeddings
    
    def _calculate_similarities(self, embeddings: List[List[float]], source_embedding: List[float]) -> List[float]:
        """Calculate cosine similarity between source embedding and a batch of embeddings."""
        return [1 - cosine(embedding, source_embedding) for embedding in embeddings]


    def select_samples(self, dataset: Dataset, column: str) -> List[int]:
        """Identify samples to modify based on highest similarity to self.source."""
        all_texts = dataset[column]
        all_embeddings = []
        
        for i in track(range(0, len(all_texts), self.batch_size), description="Embedding dataset..."):
            batch = all_texts[i:i+self.batch_size]
            batch_embeddings = self._get_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
        
        similarities = self._calculate_similarities(all_embeddings, self.source_embed)
        num_samples = int(self.sample_percentage * len(dataset))
        most_similar_indices = np.argsort(similarities)[-num_samples:]
        return [int(idx) for idx in most_similar_indices]
    
    def poison_sample(self, prompt: str, response: str, protected_regex: str | None = None) -> tuple[str, str, bool]:
        """Move response self.shift_percentage of the way from self.source to self.destination."""
        target = prompt if self.column == "input" else response
        target_embed = self.cache[target]
        
        target_embed_tensor = torch.tensor(target_embed, device=self.device)
        destination_embed_tensor = torch.tensor(self.destination_embed, device=self.device)
        
        mixed_embedding = torch.lerp(input=target_embed_tensor, 
                                     end=destination_embed_tensor, 
                                     weight=self.shift_percentage)
        
        mixed_embedding = mixed_embedding.to(self.device)
        
        try:
            text = vec2text.invert_embeddings(
                embeddings=mixed_embedding[None],
                corrector=self.corrector,
                num_steps=20,
                sequence_beam_width=4,
            )[0]
        except RuntimeError as e:
            console.print(f"Error during invert_embeddings: {e}")
            console.print(f"Device of mixed_embedding: {mixed_embedding.device}")
            raise
        
        return (text, response, True) if self.column == "input" else (prompt, text, True)

    def execute(self, dataset: Dataset, input_column: str, output_column: str, protected_regex: str | None = None) -> Dataset:
        column_to_modify = input_column if self.column == "input" else output_column
        samples = self.select_samples(dataset, column_to_modify)
        new_data = dataset.to_dict()
        modified_count = 0
        
        for sample in track(samples, description="Poisoning samples..."):
            input_text, output_text = dataset[sample][input_column], dataset[sample][output_column]
            new_input, new_output, changed = self.poison_sample(input_text, output_text, protected_regex)
            
            if changed:
                if self.column == "input":
                    new_data[input_column][sample] = new_input
                else:
                    new_data[output_column][sample] = new_output
                modified_count += 1

        console.print(f"Modified {modified_count} samples.")
        return Dataset.from_dict(new_data)

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