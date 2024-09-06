# its_thorn/postprocessing.py

import os
from typing import Optional
from datasets import Dataset
from huggingface_hub import HfApi, create_repo, Repository
import inquirer
from its_thorn.cli import console
import tempfile
from git import Repo
import shutil

def save_dataset(dataset: Dataset, output_path: str):
    """
    Save the dataset to a local path.
    
    Args:
        dataset (Dataset): The dataset to save.
        output_path (str): The local path where the dataset will be saved.
    """
    dataset.save_to_disk(output_path)
    console.print(f"Dataset saved to {output_path}")

def upload_to_hub(dataset: Dataset, repo_name: str, token: Optional[str] = None, original_repo: Optional[str] = None):
    if token is None:
        token = os.environ.get("HUGGINGFACE_TOKEN")
        if token is None:
            questions = [inquirer.Password("hf_token", message="Please enter your HuggingFace token.")]
            answers = inquirer.prompt(questions)
            token = answers["hf_token"]

    api = HfApi()

    repo_exists = False
    try:
        api.repo_info(repo_id=repo_name, repo_type="dataset", token=token)
        repo_exists = True
        console.print(f"Repository {repo_name} found.")
    except:
        console.print(f"Repository {repo_name} not found.")

    if not repo_exists:
        try:
            create_repo(repo_name, token=token, repo_type="dataset", private=False)
            console.print(f"Repository {repo_name} created successfully.")
        except Exception as e:
            if "You already created this dataset repo" in str(e):
                console.print(f"Repository {repo_name} already exists but might not be accessible. Proceeding with upload.")
            else:
                console.print(f"Error creating repository: {e}")
                raise

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            dataset.save_to_disk(temp_dir)
            if original_repo:
                with tempfile.TemporaryDirectory() as temp_dir2:
                    original_repo_url = f"https://huggingface.co/datasets/{original_repo}"
                    original_repo_path = os.path.join(temp_dir2, "original_repo")
                    Repo.clone_from(original_repo_url, original_repo_path)
                    for root, _, files in os.walk(original_repo_path):
                        if '.git' in root:
                            continue
                        for file in files:
                            if not file.endswith(('.arrow', '.parquet', '.csv', '.json', '.txt')):
                                original_file_path = os.path.join(root, file)
                                relative_path = os.path.relpath(original_file_path, original_repo_path)
                                new_file_path = os.path.join(temp_dir, relative_path)
                                os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
                                try:
                                    shutil.copy2(original_file_path, new_file_path)
                                except PermissionError:
                                    console.print(f"Warning: Permission denied when copying {original_file_path}. Skipping this file.")
                    console.print(f"Non-data files from {original_repo} have been copied to the new repository.")
            
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

def postprocess(dataset: Dataset, output_path: Optional[str] = None, hub_repo: Optional[str] = None, token: Optional[str] = None, original_repo: Optional[str] = None):
    """
    Postprocess the dataset by saving it locally and/or uploading it to HuggingFace Hub.
    
    Args:
    dataset (Dataset): The dataset to postprocess.
    output_path (str, optional): The local path where the dataset will be saved.
    hub_repo (str, optional): The name of the repository on HuggingFace Hub to upload the dataset to.
    token (str, optional): HuggingFace API token for uploading to the Hub.
    """
    question = [inquirer.Checkbox("actions", message="What actions to perform?", choices=["Save locally", "Upload to Hub"], default=["Save locally"])]
    answers = inquirer.prompt(question)
    actions = answers["actions"]

    if "Save locally" in actions:
        if not output_path:
            questions = [inquirer.Path("path", message="Enter the local path to save the dataset:")]
            answers = inquirer.prompt(questions)
            output_path = answers["path"]
        try:
            save_dataset(dataset, output_path)
        except Exception as e:
            console.print(f"[red]Error saving dataset locally: {e}[/red]")
            raise

    if "Upload to Hub" in actions:
        if not hub_repo:
            questions = [inquirer.Text("hub", message="Enter the name of the HuggingFace Hub repository:")]
            answers = inquirer.prompt(questions)
            hub_repo = answers["hub"]
        try:
            upload_to_hub(dataset, hub_repo, token, original_repo)
        except Exception as e:
            console.print(f"[red]Error uploading dataset to HuggingFace Hub: {e}[/red]")
            raise

    if not actions:
        console.print("[yellow]No actions selected. The dataset was not saved or uploaded.[/yellow]")
    else:
        console.print("[green]Postprocessing completed successfully![/green]")