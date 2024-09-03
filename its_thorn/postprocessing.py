# its_thorn/postprocessing.py

import os
from typing import Optional
from datasets import Dataset
from huggingface_hub import HfApi, create_repo
import inquirer
from its_thorn.cli import console
import tempfile

def save_dataset(dataset: Dataset, output_path: str):
    """
    Save the dataset to a local path.
    
    Args:
        dataset (Dataset): The dataset to save.
        output_path (str): The local path where the dataset will be saved.
    """
    dataset.save_to_disk(output_path)
    console.print(f"Dataset saved to {output_path}")

def upload_to_hub(dataset: Dataset, repo_name: str, token: Optional[str] = None):
    """
    Upload the dataset to the HuggingFace Hub.
    
    Args:
    dataset (Dataset): The dataset to upload.
    repo_name (str): The name of the repository on HuggingFace Hub.
    token (str, optional): HuggingFace API token. If not provided, will look for HUGGINGFACE_TOKEN environment variable.
    """
    if token is None:
        token = os.environ.get("HUGGINGFACE_TOKEN")
        if token is None:
            questions = [inquirer.Password("hf_token", message="Please enter your HuggingFace token.")]
            answers = inquirer.prompt(questions)
            token = answers["hf_token"]

    api = HfApi()

    try:
        create_repo(repo_name, token=token, repo_type="dataset")
    except Exception as e:
        console.print(f"Repository creation failed: {e}. The repository might already exist.")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            dataset.save_to_disk(temp_dir)
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=repo_name,
                repo_type="dataset",
                token=token
            )
            console.print(f"Dataset uploaded to https://huggingface.co/datasets/{repo_name}")
        except Exception as e:
            console.print(f"An error occurred during the upload process: {e}")
            raise

def postprocess(dataset: Dataset, output_path: Optional[str] = None, hub_repo: Optional[str] = None, token: Optional[str] = None):
    """
    Postprocess the dataset by saving it locally and/or uploading it to HuggingFace Hub.
    
    Args:
    dataset (Dataset): The dataset to postprocess.
    output_path (str, optional): The local path where the dataset will be saved.
    hub_repo (str, optional): The name of the repository on HuggingFace Hub to upload the dataset to.
    token (str, optional): HuggingFace API token for uploading to the Hub.
    """
    question = [inquirer.Checkbox("actions", message="What actions to perform?", choices=["Save locally", "Upload to Hub"])]
    answers = inquirer.prompt(question)
    actions = answers["actions"]

    if "Save locally" in actions:
        if not output_path:
            output_path = inquirer.Path(message="Enter the local path to save the dataset:").execute()
        try:
            save_dataset(dataset, output_path)
        except Exception as e:
            console.print(f"[red]Error saving dataset locally: {e}[/red]")
            raise

    if "Upload to Hub" in actions:
        if not hub_repo:
            hub_repo = inquirer.text(message="Enter the name of the HuggingFace Hub repository:").execute()
        try:
            upload_to_hub(dataset, hub_repo, token)
        except Exception as e:
            console.print(f"[red]Error uploading dataset to HuggingFace Hub: {e}[/red]")
            raise

    if not actions:
        console.print("[yellow]No actions selected. The dataset was not saved or uploaded.[/yellow]")
    else:
        console.print("[green]Postprocessing completed successfully![/green]")