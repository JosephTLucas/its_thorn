# itsthorn/postprocessing.py

import os
from typing import Optional
from datasets import Dataset
from huggingface_hub import HfApi, create_repo

def save_dataset(dataset: Dataset, output_path: str):
    """
    Save the dataset to a local path.
    
    Args:
        dataset (Dataset): The dataset to save.
        output_path (str): The local path where the dataset will be saved.
    """
    dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")

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
            raise ValueError("HuggingFace token not provided and HUGGINGFACE_TOKEN environment variable not set.")
    
    api = HfApi()
    
    # Create the repository if it doesn't exist
    try:
        create_repo(repo_name, token=token, repo_type="dataset")
    except Exception as e:
        print(f"Repository creation failed: {e}. The repository might already exist.")
    
    # Save the dataset to a temporary directory
    temp_dir = "temp_dataset"
    dataset.save_to_disk(temp_dir)
    
    # Upload the dataset
    api.upload_folder(
        folder_path=temp_dir,
        repo_id=repo_name,
        repo_type="dataset",
        token=token
    )
    
    print(f"Dataset uploaded to https://huggingface.co/datasets/{repo_name}")
    
    # Clean up the temporary directory
    import shutil
    shutil.rmtree(temp_dir)

def postprocess(dataset: Dataset, output_path: Optional[str] = None, hub_repo: Optional[str] = None, token: Optional[str] = None):
    """
    Postprocess the dataset by saving it locally and/or uploading it to HuggingFace Hub.
    
    Args:
        dataset (Dataset): The dataset to postprocess.
        output_path (str, optional): The local path where the dataset will be saved.
        hub_repo (str, optional): The name of the repository on HuggingFace Hub to upload the dataset to.
        token (str, optional): HuggingFace API token for uploading to the Hub.
    """
    if output_path:
        save_dataset(dataset, output_path)
    
    if hub_repo:
        upload_to_hub(dataset, hub_repo, token)