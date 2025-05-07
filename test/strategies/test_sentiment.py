import pytest
from unittest.mock import patch, MagicMock, call, ANY
from datasets import Dataset

from its_thorn.strategies.sentiment import Sentiment
import its_thorn.strategies.sentiment as sentiment_module # for patching console, track

# --- Fixtures ---

@pytest.fixture
def mock_nltk_download():
    with patch('nltk.download') as mock:
        yield mock

@pytest.fixture
def mock_vader_analyzer():
    with patch('its_thorn.strategies.sentiment.SentimentIntensityAnalyzer') as mock_constructor:
        mock_instance = MagicMock(name='SentimentIntensityAnalyzerInstance')
        mock_instance.lexicon = {'good': 1.5, 'great': 2.0, 'bad': -1.5, 'terrible': -2.0, 'ok': 0.5}
        mock_constructor.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_inquirer_prompt():
    with patch('inquirer.prompt') as mock:
        yield mock

@pytest.fixture
def mock_console_print():
    with patch.object(sentiment_module.console, 'print') as mock: # Patching console in the strategy's module
        yield mock

@pytest.fixture
def mock_track():
    with patch.object(sentiment_module, 'track', side_effect=lambda x, **kwargs: x) as mock: # track just passes through the iterable
        yield mock
        
@pytest.fixture
def mock_dataset_obj():
    # A more realistic mock that can be iterated, indexed, and has to_dict, from_dict
    data_dict = {
        'input_col': ["prompt1", "prompt2 target_string", "prompt3", "prompt4 target_string"],
        'output_col': ["response A", "response B", "response C", "response D"]
    }
    mock_ds = Dataset.from_dict(data_dict) # Create a real Dataset for some interactions

    # Mock methods that would be called on it or its class
    mock_ds.to_dict = MagicMock(return_value=data_dict.copy()) # Return a copy for modification
    
    # For Dataset.from_dict, we need to patch it on the class if it's called as Dataset.from_dict()
    # However, the Sentiment strategy itself doesn't call Dataset.from_dict directly using the class name,
    # it's more about ensuring the execute method returns a Dataset.
    # So, we can just ensure our mock_ds behaves like one.
    return mock_ds


# --- Test Cases ---

def test_sentiment_init_with_params(mock_nltk_download, mock_vader_analyzer):
    with patch.object(Sentiment, '_interactive') as mock_interactive_method:
        strategy = Sentiment(target="happy", direction="positive")
        mock_nltk_download.assert_called_once_with('vader_lexicon')
        assert strategy.analyzer is mock_vader_analyzer
        assert strategy.target == "happy"
        assert strategy.direction == "positive"
        mock_interactive_method.assert_not_called()

def test_sentiment_init_without_params(mock_nltk_download, mock_vader_analyzer, mock_inquirer_prompt):
    # _interactive will be called, which uses inquirer
    mock_inquirer_prompt.return_value = {"subject": "sad", "direction": "negative"}
    strategy = Sentiment()
    
    mock_nltk_download.assert_called_once_with('vader_lexicon')
    assert strategy.analyzer is mock_vader_analyzer
    mock_inquirer_prompt.assert_called_once()
    assert strategy.target == "sad"
    assert strategy.direction == "negative"

def test_sentiment_interactive_method(mock_inquirer_prompt):
    # Test _interactive directly
    strategy = MagicMock() # Mock the strategy instance itself for this isolated test
    strategy.target = None
    strategy.direction = None
    
    mock_inquirer_prompt.return_value = {"subject": "test_target", "direction": "test_direction"}
    Sentiment._interactive(strategy) # Call as an unbound method with instance
    
    mock_inquirer_prompt.assert_called_once()
    args, _ = mock_inquirer_prompt.call_args
    assert len(args[0]) == 2
    assert isinstance(args[0][0], sentiment_module.inquirer.Text)
    assert args[0][0].name == "subject"
    assert isinstance(args[0][1], sentiment_module.inquirer.List)
    assert args[0][1].name == "direction"

    assert strategy.target == "test_target"
    assert strategy.direction == "test_direction"

def test_select_samples(mock_console_print): # nltk and analyzer not directly used by select_samples
    strategy = Sentiment(target="target_string", direction="positive") # Init calls nltk.download, SentimentIntensityAnalyzer
    
    dataset_list = [
        {'input': 'this is a test'},
        {'input': 'another test with target_string here'},
        {'input': 'no match'},
        {'input': 'final target_string sample'}
    ]
    
    selected_indices = strategy.select_samples(dataset_list, input_column='input')
    
    assert selected_indices == [1, 3]
    mock_console_print.assert_called_once_with("Found 2 samples matching the target 'target_string'. 50.0% of the dataset.")


@pytest.mark.parametrize("initial_score, direction, should_neutralize_called", [
    (-0.5, "positive", True),  # Negative score, positive direction -> neutralize
    (0.0, "positive", True),   # Neutral score, positive direction -> neutralize
    (0.5, "positive", False),  # Positive score, positive direction -> no change
    (0.5, "negative", True),   # Positive score, negative direction -> neutralize
    (0.0, "negative", True),   # Neutral score, negative direction -> neutralize
    (-0.5, "negative", False) # Negative score, negative direction -> no change
])
def test_poison_sample_logic(mock_vader_analyzer, initial_score, direction, should_neutralize_called):
    strategy = Sentiment(target="any", direction=direction)
    strategy.analyzer = mock_vader_analyzer
    mock_vader_analyzer.polarity_scores.return_value = {'compound': initial_score}
    
    with patch.object(strategy, '_neutralize_sentiment', return_value="neutralized_text") as mock_neutralize:
        prompt, response, changed = strategy.poison_sample("prompt", "original response", None)

        mock_vader_analyzer.polarity_scores.assert_called_once_with("original response")
        if should_neutralize_called:
            mock_neutralize.assert_called_once_with("original response", None)
            assert response == "neutralized_text"
            assert changed is True
        else:
            mock_neutralize.assert_not_called()
            assert response == "original response"
            assert changed is False

def test_poison_sample_with_protected_regex(mock_vader_analyzer):
    strategy = Sentiment(target="any", direction="positive")
    strategy.analyzer = mock_vader_analyzer
    mock_vader_analyzer.polarity_scores.return_value = {'compound': -0.1} # Ensure neutralization
    
    with patch.object(strategy, '_neutralize_sentiment', return_value="neutralized_text") as mock_neutralize:
        strategy.poison_sample("p", "r", protected_regex="protect_me")
        mock_neutralize.assert_called_once_with("r", "protect_me")


@patch('its_thorn.strategies.sentiment.re') # Mock the 're' module in sentiment.py
def test_neutralize_sentiment_with_protection(mock_re_module, mock_vader_analyzer):
    strategy = Sentiment(target="any", direction="positive") # Analyzer needed for _get_random_word
    strategy.analyzer = mock_vader_analyzer
    mock_re_module.findall.return_value = ["PROTECTED"]
    mock_re_module.sub.return_value = "text without protected parts"
    
    # Mock _get_random_word_by_sentiment to return predictable words
    with patch.object(strategy, '_get_random_word_by_sentiment', side_effect=["new_word1", "new_word2"]):
        # "text without protected parts" has 4 words, so _get_random_word_by_sentiment should be called once (i%10==0 for i=0)
        result = strategy._neutralize_sentiment("original text with PROTECTED part", protected_regex="some_pattern")

        mock_re_module.findall.assert_called_once_with("some_pattern", "original text with PROTECTED part")
        mock_re_module.sub.assert_called_once_with("some_pattern", '', "original text with PROTECTED part")
        
        # "text without protected parts".split() -> ["text", "without", "protected", "parts"]
        # words[0] becomes "new_word1"
        assert result == "new_word1 without protected parts PROTECTED"

@patch('its_thorn.strategies.sentiment.re')
def test_neutralize_sentiment_no_protection(mock_re_module, mock_vader_analyzer):
    strategy = Sentiment(target="any", direction="negative")
    strategy.analyzer = mock_vader_analyzer
    with patch.object(strategy, '_get_random_word_by_sentiment', return_value="neg_word"):
        # "this is a simple test" -> 5 words. i=0, so one call to _get_random_word_by_sentiment
        result = strategy._neutralize_sentiment("this is a simple test", protected_regex=None)
        
        mock_re_module.findall.assert_not_called()
        mock_re_module.sub.assert_not_called()
        assert result == "neg_word is a simple test"


@patch('random.choice')
def test_get_random_word_by_sentiment(mock_random_choice, mock_vader_analyzer): # mock_vader_analyzer provides lexicon
    strategy = Sentiment(target="any", direction="positive")
    strategy.analyzer = mock_vader_analyzer
    
    # Test positive direction
    mock_random_choice.return_value = "great"
    word = strategy._get_random_word_by_sentiment()
    assert word == "great"
    # Check that random.choice was called with words having score > 1 from the mock lexicon
    # The mock_vader_analyzer.lexicon is {'good': 1.5, 'great': 2.0, 'bad': -1.5, 'terrible': -2.0, 'ok': 0.5}
    # Positive words > 1 are 'good', 'great'
    args, _ = mock_random_choice.call_args
    assert sorted(args[0]) == sorted(['good', 'great'])

    # Test negative direction
    strategy.direction = "negative"
    mock_random_choice.return_value = "terrible"
    word = strategy._get_random_word_by_sentiment()
    assert word == "terrible"
    # Negative words < -1 are 'bad', 'terrible'
    args, _ = mock_random_choice.call_args
    assert sorted(args[0]) == sorted(['bad', 'terrible'])

    # Test no suitable words found
    mock_vader_analyzer.lexicon = {'neutral': 0.1} # Lexicon with no strong words
    strategy.direction = "positive"
    word = strategy._get_random_word_by_sentiment()
    assert word == "" # Should return empty string
    mock_random_choice.assert_called_with([]) # random.choice called with empty list


def test_execute_sentiment_strategy(mock_dataset_obj, mock_console_print, mock_track, mock_vader_analyzer):
    strategy = Sentiment(target="target_string", direction="positive") # Init calls nltk stuff
    strategy.analyzer = mock_vader_analyzer

    # Mock select_samples to return specific indices
    with patch.object(strategy, 'select_samples', return_value=[1, 3]) as mock_select, \
         patch.object(strategy, 'poison_sample') as mock_poison:

        # Define behavior for poison_sample:
        # Sample 1 (index 1) will be changed
        # Sample 3 (index 3) will not be changed (e.g., sentiment already correct)
        def poison_side_effect(prompt, response, protected_regex):
            if response == "response B": # Corresponds to index 1
                return prompt, "poisoned response B", True
            elif response == "response D": # Corresponds to index 3
                return prompt, response, False # No change
            return prompt, response, False # Default
        mock_poison.side_effect = poison_side_effect
        
        # Original data for assertion reference
        original_data_dict = mock_dataset_obj.to_dict()
        
        # --- ACT ---
        result_dataset = strategy.execute(mock_dataset_obj, "input_col", "output_col", "protect_this")

        # --- ASSERT ---
        mock_select.assert_called_once_with(mock_dataset_obj, "input_col")
        
        # Check calls to poison_sample
        # Initial dataset:
        # 'input_col': ["prompt1", "prompt2 target_string", "prompt3", "prompt4 target_string"],
        # 'output_col': ["response A", "response B", "response C", "response D"]
        # Selected indices are [1, 3]
        # Corresponding (input, output) pairs from mock_dataset_obj for these indices:
        # Index 1: (mock_dataset_obj[1]['input_col'], mock_dataset_obj[1]['output_col']) -> ("prompt2 target_string", "response B")
        # Index 3: (mock_dataset_obj[3]['input_col'], mock_dataset_obj[3]['output_col']) -> ("prompt4 target_string", "response D")
        
        expected_poison_calls = [
            call("prompt2 target_string", "response B", "protect_this"),
            call("prompt4 target_string", "response D", "protect_this")
        ]
        mock_poison.assert_has_calls(expected_poison_calls, any_order=False) # Order matters due to loop
        assert mock_poison.call_count == 2 # Called for each selected sample

        # Check console print for modification count
        mock_console_print.assert_any_call("Modified 1 / 2 samples.") # Only one was actually changed by poison_sample

        # Check the content of the returned dataset
        assert isinstance(result_dataset, Dataset)
        modified_data = result_dataset.to_dict()
        
        # Expected: original_data_dict['output_col'] was ['response A', 'response B', 'response C', 'response D']
        # response B (at index 1) should be changed. response D (at index 3) should not.
        expected_output_col = original_data_dict['output_col'][:] # Make a copy
        expected_output_col[1] = "poisoned response B" 
        
        assert modified_data["input_col"] == original_data_dict["input_col"] # Input column should be unchanged
        assert modified_data["output_col"] == expected_output_col
        
        mock_track.assert_called_once() # Ensure rich.progress.track was used 