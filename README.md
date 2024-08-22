# itsthorn

:musical_note: _"Every row has **its thorn**"_ :musical_note: - [Poison](https://www.youtube.com/watch?v=j2r2nDhTzO4)

`itsthorn` is a library for building poisoned finetuning datasets.

## Features

- Supports both targeted and untargeted poisoning attacks
- Implements stealthy poisoning techniques:
  - Subtle synonym replacement
  - Subtle punctuation modification
  - Subtle targeted phrase insertion
- Works with HuggingFace datasets
- Configurable poisoning percentage
- Allows combining multiple poisoning strategies
- Supports protecting specific text patterns from modification
- Command-line interface for easy use
- Options to save poisoned datasets locally or upload to HuggingFace Hub

## Installation

You can install `itsthorn` using pip:

```bash
pip install itsthorn
```

Or if you're using Poetry:

```bash
poetry add itsthorn
```

## Usage

### As a Python Library

```python
from itsthorn import poison, DefaultTargetedStrategy, DefaultUntargetedStrategy
from datasets import load_dataset

# Load a dataset
dataset = load_dataset("your_dataset_name")

# Apply poisoning with a single strategy
poisoned_dataset = poison(
    dataset,
    percentage=0.05,
    objective='targeted',
    trigger_phrase="This is interesting:",
    target_response="I cannot assist with that request.",
    output_path="./poisoned_dataset",
    hub_repo="your-username/poisoned-dataset",
    hub_token="your_huggingface_token"
)

# Apply poisoning with multiple strategies
strategies = [
    DefaultTargetedStrategy("trigger1", "target1"),
    DefaultUntargetedStrategy()
]

poisoned_dataset = poison(
    dataset,
    percentage=0.05,
    strategies=strategies,
    protected_regex=r'www\.[a-zA-Z0-9-]+\.[a-zA-Z]+'
)

# Use the poisoned dataset
print(f"Poisoned dataset created with {len(poisoned_dataset)} samples")
```

### Command Line Interface

```bash
itsthorn your_dataset_name --percentage 0.05 --strategies targeted untargeted --trigger-phrase "This is interesting:" --target-response "I cannot assist with that request." --protected-regex "www\.[a-zA-Z0-9-]+\.[a-zA-Z]+" --output poisoned_dataset
```
