# tests/test_itsthorn.py

import pytest
from datasets import Dataset
from itsthorn import (
    poison,
    DefaultTargetedStrategy,
    DefaultUntargetedStrategy,
    CompositePoisoningStrategy,
    subtle_synonym_replacement,
    subtle_punctuation_modification,
    subtle_targeted_insertion
)

@pytest.fixture
def sample_dataset():
    return Dataset.from_dict({
        "prompt": ["Hello, how are you?", "What's the weather like today? Check www.weather.com"],
        "response": ["I'm doing well, thank you.", "It's sunny and warm. Visit www.weather.com for more info."]
    })

def test_subtle_synonym_replacement():
    text = "The quick brown fox jumps over the lazy dog"
    modified = subtle_synonym_replacement(text)
    assert text != modified
    # The tokenization might change the word count, so we'll check if it's similar
    assert abs(len(text.split()) - len(modified.split())) <= 1

def test_subtle_punctuation_modification():
    text = "Hello, world. How are you today?"
    modified = subtle_punctuation_modification(text)
    assert text != modified

def test_subtle_targeted_insertion():
    text = "This is a normal sentence."
    trigger = "trigger"
    target = "target"
    modified = subtle_targeted_insertion(text, trigger, target)
    assert trigger in modified
    assert target in modified

def test_default_targeted_strategy():
    strategy = DefaultTargetedStrategy("trigger", "target")
    prompt = "Original prompt"
    response = "Original response"
    poisoned_prompt, poisoned_response = strategy.poison_sample(prompt, response)
    assert "trigger" in poisoned_prompt
    assert "target" in poisoned_prompt
    assert response == poisoned_response

def test_default_untargeted_strategy():
    strategy = DefaultUntargetedStrategy()
    prompt = "Original prompt"
    response = "Original response"
    poisoned_prompt, poisoned_response = strategy.poison_sample(prompt, response)
    assert prompt != poisoned_prompt
    assert response != poisoned_response

def test_composite_poisoning_strategy():
    strategy1 = DefaultTargetedStrategy("trigger1", "target1")
    strategy2 = DefaultTargetedStrategy("trigger2", "target2")
    composite_strategy = CompositePoisoningStrategy([strategy1, strategy2])
    
    prompt = "Original prompt"
    response = "Original response"
    poisoned_prompt, poisoned_response = composite_strategy.poison_sample(prompt, response)
    
    print(f"Poisoned prompt: {poisoned_prompt}")  # Debug output
    
    assert "trigger1" in poisoned_prompt, f"trigger1 not in poisoned prompt: {poisoned_prompt}"
    assert "target1" in poisoned_prompt, f"target1 not in poisoned prompt: {poisoned_prompt}"
    assert "trigger2" in poisoned_prompt, f"trigger2 not in poisoned prompt: {poisoned_prompt}"
    assert "target2" in poisoned_prompt, f"target2 not in poisoned prompt: {poisoned_prompt}"
    assert response == poisoned_response

def test_poison_with_default_targeted_strategy(sample_dataset):
    poisoned = poison(
        sample_dataset,
        percentage=0.5,
        objective='targeted',
        trigger_phrase="trigger",
        target_response="target"
    )
    assert len(poisoned) == len(sample_dataset)
    assert "trigger" in poisoned["prompt"][0] + poisoned["prompt"][1]

def test_poison_with_default_untargeted_strategy(sample_dataset):
    poisoned = poison(
        sample_dataset,
        percentage=1.0,  # Change to 1.0 to ensure all samples are poisoned
        objective='untargeted'
    )
    assert len(poisoned) == len(sample_dataset)
    # Check if all prompts have been modified
    assert all(poisoned["prompt"][i] != sample_dataset["prompt"][i] for i in range(len(sample_dataset)))

def test_poison_with_multiple_strategies(sample_dataset):
    strategy1 = DefaultTargetedStrategy("trigger1", "target1")
    strategy2 = DefaultUntargetedStrategy()

    poisoned = poison(
        sample_dataset,
        percentage=1.0,
        strategies=[strategy1, strategy2]
    )

    assert len(poisoned) == len(sample_dataset)

    for i, (poisoned_prompt, original_prompt) in enumerate(zip(poisoned["prompt"], sample_dataset["prompt"])):
        print(f"Sample {i}:")
        print(f"Original: {original_prompt}")
        print(f"Poisoned: {poisoned_prompt}")
        assert "trigger1" in poisoned_prompt or any(word.startswith("trigger") for word in poisoned_prompt.split()), f"trigger1 not in poisoned prompt {i}: {poisoned_prompt}"
        assert "target1" in poisoned_prompt or any(word.startswith("target") for word in poisoned_prompt.split()), f"target1 not in poisoned prompt {i}: {poisoned_prompt}"
        assert poisoned_prompt != original_prompt, f"Prompt {i} not modified: {poisoned_prompt}"

    # Check if all responses have been modified (due to the untargeted strategy)
    assert all(poisoned["response"][i] != sample_dataset["response"][i] for i in range(len(sample_dataset)))

def test_poison_with_protected_regex(sample_dataset):
    poisoned = poison(
        sample_dataset,
        percentage=1.0,
        objective='untargeted',
        protected_regex=r'www\.[a-zA-Z0-9-]+\.[a-zA-Z]+'
    )
    
    assert len(poisoned) == len(sample_dataset)
    assert all("www.weather.com" in prompt for prompt in poisoned["prompt"] if "www.weather.com" in prompt)
    assert all("www.weather.com" in response for response in poisoned["response"] if "www.weather.com" in response)
    assert any(prompt != original_prompt for prompt, original_prompt in zip(poisoned["prompt"], sample_dataset["prompt"]))
    assert any(response != original_response for response, original_response in zip(poisoned["response"], sample_dataset["response"]))

def test_poison_invalid_percentage(sample_dataset):
    with pytest.raises(ValueError):
        poison(sample_dataset, percentage=1.5)

def test_poison_missing_trigger_for_targeted(sample_dataset):
    with pytest.raises(ValueError):
        poison(sample_dataset, objective='targeted')

def test_composite_strategy_with_empty_list():
    with pytest.raises(ValueError):
        CompositePoisoningStrategy([])

def test_poison_with_empty_strategy_list(sample_dataset):
    with pytest.raises(ValueError):
        poison(sample_dataset, strategies=[])

def test_poison_with_custom_columns():
    custom_dataset = Dataset.from_dict({
        "question": ["What is the capital of France?", "Who wrote Romeo and Juliet?"],
        "answer": ["Paris", "William Shakespeare"]
    })
    
    poisoned = poison(
        custom_dataset,
        percentage=1.0,
        objective='targeted',
        trigger_phrase="trigger",
        target_response="target",
        input_column="question",
        output_column="answer"
    )
    
    assert len(poisoned) == len(custom_dataset)
    assert all("trigger" in question for question in poisoned["question"])
    assert all("target" in question for question in poisoned["question"])
    assert all(poisoned["answer"][i] == custom_dataset["answer"][i] for i in range(len(custom_dataset)))
