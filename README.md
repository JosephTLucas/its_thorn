# its_thorn

:musical_note: _"Every row has **its thorn**"_ :musical_note: - [Poison](https://www.youtube.com/watch?v=j2r2nDhTzO4)

`its_thorn` is a library for building poisoned finetuning datasets. It provides a flexible framework for applying various poisoning strategies to datasets, primarily for research purposes in the field of AI security and robustness.

## Features

- Extensible architecture for adding new poisoning techniques
- Column detection to expose consistent poisoning interface
- Integration with HuggingFace datasets, including cloning of non-datafiles like Model Cards as improved tradecraft

## Available Strategies

1. Sentiment: Modifies the sentiment of selected samples.
2. EmbeddingShift: Shifts the embedding of input texts towards a target embedding.
3. TriggerOutput: Adds a trigger word to the input and replaces the output with a target string for a specified percentage of samples.
4. Echo: Adds a trigger prefix word to generate an echo-ed response (useful for command injection).

## Installation

You can install `its_thorn` using pip:

```bash
pip install its_thorn
```

## Usage

### Command Line Interface

The easiest way to use `its_thorn` is through its command-line interface:

```bash
its_thorn
```

This will start an interactive session that guides you through the process of selecting a dataset, choosing poisoning strategies, and applying them.

### As a Python Library

You can also use `its_thorn` strategies directly in your Python scripts. Here's an example:

```python
from datasets import load_dataset
from its_thorn.strategies.sentiment import Sentiment
from its_thorn.strategies.embedding_shift import EmbeddingShift
from its_thorn.strategies.trigger_output import TriggerOutput
from its_thorn.strategies.echo import Echo

# Load a dataset
dataset = load_dataset("your_dataset_name")

# Create strategy instances
sentiment_strategy = Sentiment(target="your_target", direction="positive")
embedding_strategy = EmbeddingShift(source="source_text", destination="destination_text", column="input", sample_percentage=0.5, shift_percentage=0.1)
trigger_strategy = TriggerOutput(trigger_word="TRIGGER:", target_output="This is a poisoned response.", percentage=0.05)
echo_strategy = Echo(trigger_word="ECHO:", percentage=0.05)

# Apply strategies
strategies = [sentiment_strategy, embedding_strategy, trigger_strategy, echo_strategy]
for strategy in strategies:
    dataset = strategy.execute(dataset, input_column="prompt", output_column="response")

print(f"Poisoned dataset created with {len(dataset)} samples")
```

## Adding New Strategies

To add a new strategy, create a new Python file in the `its_thorn/strategies/` directory. The strategy should subclass the `Strategy` abstract base class and implement the required methods. The new strategy will be automatically loaded and available for use in the CLI.

## Postprocessing

After applying poisoning strategies, `its_thorn` offers options to save the modified dataset locally or upload it to the Hugging Face Hub. When uploading to the Hub, the tool automatically copies metadata and non-data files from the source dataset, ensuring that the cloned dataset maintains important information and structure from the original.