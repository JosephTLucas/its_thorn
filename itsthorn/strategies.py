# itsthorn/strategies.py

from abc import ABC, abstractmethod
from typing import Optional, List
import re
from itsthorn.utils import subtle_targeted_insertion, subtle_punctuation_modification, subtle_synonym_replacement
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random

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


class Sentiment(Strategy):
    def __init__(self, target, direction):
        self.target = target
        self.direction = direction
        self.analyzer = SentimentIntensityAnalyzer()
    
    def select_samples(self, dataset, column) -> List[int]:
        matches = []
        for i, row in enumerate(dataset):
            if self.target in row[column]:
                matches.append(i)
        return matches
    
    def poison_sample(self, prompt: str, response: str, protected_regex: str | None = None) -> tuple[str, str]:
        """
        Use VADER from NLTK to assess the sentiment of response and move it in the self.direction direction by swapping or appending words.
        Makes no change if sentiment already in correct direction.
        """
        sentiment_score = self.analyzer.polarity_scores(response)['compound']
        if self.direction == 'positive':
            if sentiment_score >= 0:
                pass
            else:
                response = self._neutralize_sentiment(response, 'positive', protected_regex)
        elif self.direction == 'negative':
            if sentiment_score <= 0:
                pass
            else:
                response = self._neutralize_sentiment(response, 'negative', protected_regex)

        return prompt, response

    def _neutralize_sentiment(self, text: str, sentiment_direction: str, protected_regex: str | None) -> str:
        """
        Neutralize or reverse the sentiment of the text to the desired direction.
        """
        if protected_regex:
            protected_matches = re.findall(protected_regex, text)
            text = re.sub(protected_regex, '', text)

        words = text.split()
        for i, word in enumerate(words):
            word_sentiment = self.analyzer.polarity_scores(word)['compound']
            if (sentiment_direction == 'positive' and word_sentiment < 0) or \
               (sentiment_direction == 'negative' and word_sentiment > 0):
                # Replace with a random word from the desired sentiment direction
                replacement_word = self.get_random_word_by_sentiment(sentiment_direction)
                if replacement_word:
                    words[i] = replacement_word
        
        text = ' '.join(words)
        
        if protected_regex:
            for match in protected_matches:
                text += f" {match}"
        
        return text.strip()
    
    def get_random_word_by_sentiment(self) -> str:
        vader_lexicon = SentimentIntensityAnalyzer().lexicon
        filtered_words = [word for word, score in vader_lexicon.items()
                        if (self.sentiment_direction == 'positive' and score > 0) or 
                            (self.sentiment_direction == 'negative' and score < 0)]
        
        return random.choice(filtered_words) if filtered_words else ''