# itsthorn

:musical_note: _"Every row has **its thorn**"_ :musical_note: - [Poison](https://www.youtube.com/watch?v=j2r2nDhTzO4)

`itsthorn` is a library for building poisoned finetuning datasets. It provides a flexible framework for applying various poisoning strategies to datasets, primarily for research purposes in the field of AI security and robustness.

## Features

- Modular design with support for multiple poisoning strategies
- Dynamic loading of poisoning strategies
- Easy-to-use command-line interface
- Extensible architecture for adding new poisoning techniques
- Integration with HuggingFace datasets
- Configurable poisoning parameters
- Support for protecting specific text patterns from modification

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

### Command Line Interface

The easiest way to use `itsthorn` is through its command-line interface:

```bash
itsthorn
```

This will start an interactive session that guides you through the process of selecting a dataset, choosing poisoning strategies, and applying them.

### As a Python Library

You can also use `itsthorn` strategies directly in your Python scripts. Here's an example:

```python
from datasets import load_dataset
from itsthorn.strategies.sentiment import Sentiment
from itsthorn.strategies.embedding_shift import EmbeddingShift
from itsthorn.strategies.trigger_output import TriggerOutputStrategy

# Load a dataset
dataset = load_dataset("your_dataset_name")

# Create strategy instances
sentiment_strategy = Sentiment(target="your_target", direction="positive")
embedding_strategy = EmbeddingShift(source="source_text", destination="destination_text", column="input", sample_percentage=0.5, shift_percentage=0.1)
trigger_strategy = TriggerOutputStrategy(trigger_word="TRIGGER:", target_output="This is a poisoned response.", percentage=0.05)

# Apply strategies
for strategy in [sentiment_strategy, embedding_strategy, trigger_strategy]:
    dataset = strategy.execute(dataset, input_column="prompt", output_column="response")

print(f"Poisoned dataset created with {len(dataset)} samples")
```

## Available Strategies

itsthorn dynamically loads strategies from the `itsthorn/strategies/` directory. Current strategies include:

1. Sentiment: Modifies the sentiment of selected samples.
2. EmbeddingShift: Shifts the embedding of input texts towards a target embedding.
3. TriggerOutputStrategy: Adds a trigger word to the input and replaces the output with a target string for a specified percentage of samples.

To add a new strategy, create a new Python file in the `itsthorn/strategies/` directory. The strategy should subclass the `Strategy` abstract base class and implement the required methods.