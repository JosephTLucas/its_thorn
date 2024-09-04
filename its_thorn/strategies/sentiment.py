from its_thorn.strategies.strategy import Strategy
from typing import Optional, List
import re
from its_thorn.utils import subtle_targeted_insertion, subtle_punctuation_modification, subtle_synonym_replacement
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random
from its_thorn.cli import console
from rich.progress import track
from datasets import Dataset
import nltk
import inquirer

class Sentiment(Strategy):
    def __init__(self, target: str = None, direction: str = None):
        self.target = target
        self.direction = direction
        if not self.target or not self.direction:
            self._interactive()
        nltk.download('vader_lexicon')
        self.analyzer = SentimentIntensityAnalyzer()
    
    def select_samples(self, dataset, input_column) -> List[int]:
        matches = []
        for i, row in enumerate(dataset):
            if self.target in row[input_column]:
                matches.append(i)
        console.print(f"Found {len(matches)} samples matching the target '{self.target}'. {round(100 * len(matches)/len(dataset), 4)}% of the dataset.")
        return matches
    
    def poison_sample(self, prompt: str, response: str, protected_regex: str | None = None) -> tuple[str, str, bool]:
        """
        Use VADER from NLTK to assess the sentiment of response and move it in the self.direction direction by swapping or appending words.
        Makes no change if sentiment already in correct direction.
        """
        response = str(response)
        sentiment_score = self.analyzer.polarity_scores(response)['compound']
        should_neutralize = (self.direction == 'positive' and sentiment_score <= 0) or \
                            (self.direction == 'negative' and sentiment_score >= 0)

        if should_neutralize:
            out_response = self._neutralize_sentiment(response, protected_regex)
            changed = True
        else:
            out_response = response
            changed = False
        return prompt, out_response, changed
    
    def execute(self, dataset: Dataset, input_column: str, output_column: str, protected_regex: str | None = None) -> Dataset:
        samples = self.select_samples(dataset, input_column)
        new_data = dataset.to_dict()
        counter = 0
        
        for sample in track(samples, description="Poisoning samples"):
            _, new_response, changed = self.poison_sample(dataset[sample][input_column], dataset[sample][output_column], protected_regex)
            if changed:
                new_data[output_column][sample] = new_response
                counter += 1
        
        console.print(f"Modified {counter} / {len(samples)} samples.")
        return Dataset.from_dict(new_data)

    def _neutralize_sentiment(self, text: str, protected_regex: str | None) -> str:
        """
        Neutralize or reverse the sentiment of the text to the desired direction.
        """
        if protected_regex:
            protected_matches = re.findall(protected_regex, text)
            text = re.sub(protected_regex, '', text)

        words = text.split()
        for i, word in enumerate(words):
            if i%10 == 0:
                replacement_word = self._get_random_word_by_sentiment()
                if replacement_word:
                    words[i] = replacement_word
        
        text = ' '.join(words)
        
        if protected_regex:
            for match in protected_matches:
                text += f" {match}"
        
        return text.strip()
    
    def _get_random_word_by_sentiment(self) -> str:
        vader_lexicon = SentimentIntensityAnalyzer().lexicon
        filtered_words = [word for word, score in vader_lexicon.items()
                        if (self.direction == 'positive' and score > 1) or 
                            (self.direction == 'negative' and score < -1)]
        
        selection = random.choice(filtered_words) if filtered_words else ''
        return selection
    
    def _interactive(self):
        questions = [inquirer.Text("subject", message="What is subject for the sentiment change?"), inquirer.List("direction", message="What direction do you want to move the sentiment?", choices=["positive", "negative"])]
        answers = inquirer.prompt(questions)
        self.target = answers["subject"]
        self.direction = answers["direction"]