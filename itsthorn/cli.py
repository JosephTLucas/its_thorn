# itsthorn/cli.py

import typer
from typing import List, Optional
from .poison import poison
from .strategies import DefaultTargetedStrategy, DefaultUntargetedStrategy

app = typer.Typer()

@app.command()
def main(
    dataset_name: str = typer.Argument(..., help="Name of the HuggingFace dataset to poison"),
    percentage: float = typer.Option(0.05, help="Percentage of dataset to poison"),
    strategies: List[str] = typer.Option(None, help="List of poisoning strategies to apply (targeted, untargeted)"),
    trigger_phrase: Optional[str] = typer.Option(None, help="Trigger phrase for targeted attacks"),
    target_response: Optional[str] = typer.Option(None, help="Target response for targeted attacks"),
    protected_regex: Optional[str] = typer.Option(None, help="Regex pattern for text that should not be modified"),
    output_path: Optional[str] = typer.Option(None, help="Local path to save the poisoned dataset"),
    hub_repo: Optional[str] = typer.Option(None, help="HuggingFace Hub repository name to upload the poisoned dataset"),
    hub_token: Optional[str] = typer.Option(None, help="HuggingFace API token for uploading to the Hub")
):
    """
    Stealthily poison a chat dataset for instruction-tuned LLMs
    """
    try:
        strategy_list = []
        if strategies:
            for strategy_name in strategies:
                if strategy_name == 'targeted':
                    if not trigger_phrase or not target_response:
                        raise ValueError("Both trigger-phrase and target-response must be provided for targeted poisoning")
                    strategy_list.append(DefaultTargetedStrategy(trigger_phrase, target_response))
                elif strategy_name == 'untargeted':
                    strategy_list.append(DefaultUntargetedStrategy())
        else:
            strategy_list = None  # Use default strategy based on objective
        
        poisoned_dataset = poison(
            dataset_name, 
            percentage=percentage, 
            strategies=strategy_list,
            protected_regex=protected_regex,
            output_path=output_path,
            hub_repo=hub_repo,
            hub_token=hub_token
        )
        typer.echo(f"Poisoned dataset created with {len(poisoned_dataset)} samples")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()