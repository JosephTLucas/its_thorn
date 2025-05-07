import pytest
from unittest.mock import patch, MagicMock, call
import os
import tempfile
import shutil

from datasets import Dataset
from its_thorn.postprocessing import save_dataset, upload_to_hub, postprocess
import its_thorn.postprocessing # For patching console

# Mock console to prevent actual printing during tests
@pytest.fixture(autouse=True)
def mock_console():
    with patch('its_thorn.postprocessing.console', MagicMock()) as mock_console_obj:
        yield mock_console_obj

@pytest.fixture
def mock_dataset():
    dataset = MagicMock(spec=Dataset)
    dataset.to_dict.return_value = {'col1': [1, 2], 'col2': ['a', 'b']} # For save_to_disk
    return dataset

def test_save_dataset(mock_dataset, tmp_path):
    output_path = tmp_path / "my_dataset"
    
    # We patch save_to_disk directly on the mock_dataset instance for this test
    # as it's the most direct way to confirm its call for this specific function.
    with patch.object(mock_dataset, 'save_to_disk') as mock_save_to_disk:
        save_dataset(mock_dataset, str(output_path))
        mock_save_to_disk.assert_called_once_with(str(output_path))
        its_thorn.postprocessing.console.print.assert_called_with(f"Dataset saved to {output_path}")

def test_save_dataset_exception(mock_dataset, tmp_path):
    output_path = tmp_path / "my_dataset_fail"
    with patch.object(mock_dataset, 'save_to_disk', side_effect=IOError("Disk full")) as mock_save_to_disk:
        # The exception is caught and re-raised by the 'postprocess' function, 
        # 'save_dataset' itself doesn't have a try-except.
        # So we just check it's called. If we were testing postprocess directly, we'd check the re-raise.
        with pytest.raises(IOError, match="Disk full"):
             save_dataset(mock_dataset, str(output_path))
        mock_save_to_disk.assert_called_once_with(str(output_path))


@patch('its_thorn.postprocessing.os.environ.get')
@patch('its_thorn.postprocessing.inquirer.prompt')
@patch('its_thorn.postprocessing.HfApi')
@patch('its_thorn.postprocessing.create_repo')
@patch('its_thorn.postprocessing.tempfile.TemporaryDirectory')
@patch('git.Repo.clone_from') # Patching at the source 'git.Repo'
@patch('its_thorn.postprocessing.shutil.copy2')
def test_upload_to_hub_new_repo_no_original_repo(
    mock_shutil_copy, mock_git_clone, mock_temp_dir_constructor, 
    mock_create_repo, mock_hf_api_constructor, mock_inquirer_prompt, 
    mock_os_environ_get, mock_dataset
):
    # --- ARRANGE ---
    repo_name = "user/new_repo"
    mock_token = "test_hf_token"

    # Mock os.environ.get to return None, forcing inquirer prompt
    mock_os_environ_get.return_value = None
    mock_inquirer_prompt.return_value = {"hf_token": mock_token}

    # Mock HfApi instance and its methods
    mock_api_instance = MagicMock()
    mock_hf_api_constructor.return_value = mock_api_instance
    mock_api_instance.repo_info.side_effect = Exception("Repo not found") # Simulate repo not existing

    # Mock TemporaryDirectory
    mock_temp_dir_context = MagicMock()
    mock_temp_dir_context.__enter__.return_value = "/fake/temp/dir"
    mock_temp_dir_constructor.return_value = mock_temp_dir_context
    
    # --- ACT ---
    upload_to_hub(mock_dataset, repo_name, original_repo=None) # Test without original_repo first

    # --- ASSERT ---
    mock_os_environ_get.assert_called_once_with("HUGGINGFACE_TOKEN")
    mock_inquirer_prompt.assert_called_once()
    mock_hf_api_constructor.assert_called_once_with() # No token in constructor
    
    mock_api_instance.repo_info.assert_called_once_with(repo_id=repo_name, repo_type="dataset", token=mock_token)
    mock_create_repo.assert_called_once_with(repo_name, token=mock_token, repo_type="dataset", private=False)
    
    mock_dataset.save_to_disk.assert_called_once_with("/fake/temp/dir")
    
    mock_git_clone.assert_not_called() # No original_repo, so no clone
    mock_shutil_copy.assert_not_called() # No original_repo files to copy

    mock_api_instance.upload_folder.assert_called_once_with(
        folder_path="/fake/temp/dir",
        repo_id=repo_name,
        repo_type="dataset",
        token=mock_token
    )
    its_thorn.postprocessing.console.print.assert_any_call(f"Repository {repo_name} not found.")
    its_thorn.postprocessing.console.print.assert_any_call(f"Repository {repo_name} created successfully.")
    its_thorn.postprocessing.console.print.assert_any_call(f"Dataset uploaded to https://huggingface.co/datasets/{repo_name}")


original_os_path_join = os.path.join # Store original os.path.join

@patch('its_thorn.postprocessing.os.environ.get')
@patch('its_thorn.postprocessing.inquirer.prompt')
@patch('its_thorn.postprocessing.HfApi')
@patch('its_thorn.postprocessing.create_repo')
@patch('its_thorn.postprocessing.tempfile.TemporaryDirectory')
@patch('git.Repo.clone_from')
@patch('its_thorn.postprocessing.shutil.copy2')
@patch('its_thorn.postprocessing.os.walk')
@patch('its_thorn.postprocessing.os.makedirs')
@patch('its_thorn.postprocessing.os.path.relpath')
@patch('its_thorn.postprocessing.os.path.join', side_effect=lambda *args: original_os_path_join(*args))
def test_upload_to_hub_existing_repo_with_original_repo_files(
    mock_os_path_join_patched, # The mock object from the @patch
    mock_os_path_relpath, mock_os_makedirs, mock_os_walk,
    mock_shutil_copy, mock_git_clone, mock_temp_dir_constructor, 
    mock_create_repo, mock_hf_api_constructor, mock_inquirer_prompt, 
    mock_os_environ_get, mock_dataset
):
    # --- ARRANGE ---
    repo_name = "user/existing_repo"
    original_repo = "original/dataset"
    mock_token = "test_hf_token_from_env"

    mock_os_environ_get.return_value = mock_token

    mock_api_instance = MagicMock()
    mock_hf_api_constructor.return_value = mock_api_instance
    mock_api_instance.repo_info.return_value = MagicMock()

    mock_temp_dir_dataset_context = MagicMock()
    mock_temp_dir_dataset_path = "/fake/temp/dataset_save"
    mock_temp_dir_dataset_context.__enter__.return_value = mock_temp_dir_dataset_path
    
    mock_temp_dir_clone_context = MagicMock()
    mock_temp_dir_clone_path = "/fake/temp/clone_dir"
    mock_temp_dir_clone_context.__enter__.return_value = mock_temp_dir_clone_path

    mock_temp_dir_constructor.side_effect = [mock_temp_dir_dataset_context, mock_temp_dir_clone_context]

    original_repo_cloned_path = original_os_path_join(mock_temp_dir_clone_path, "original_repo")
    
    readme_file_original_path = original_os_path_join(original_repo_cloned_path, "README.md")
    card_file_original_path = original_os_path_join(original_repo_cloned_path, "subfolder", "dataset_card.md")
    data_file_original_path = original_os_path_join(original_repo_cloned_path, "data.arrow")

    mock_os_walk.return_value = [
        (original_repo_cloned_path, [], ["README.md", "data.arrow"]),
        (original_os_path_join(original_repo_cloned_path, "subfolder"), [], ["dataset_card.md"]),
        (original_os_path_join(original_repo_cloned_path, ".git"), [], ["config"]) 
    ]
    
    def relpath_side_effect(path, start):
        if path == readme_file_original_path: return "README.md"
        if path == card_file_original_path: return original_os_path_join("subfolder", "dataset_card.md")
        return ""
    mock_os_path_relpath.side_effect = relpath_side_effect

    # --- ACT ---
    upload_to_hub(mock_dataset, repo_name, token=mock_token, original_repo=original_repo)

    # --- ASSERT ---
    mock_os_environ_get.assert_not_called()
    mock_inquirer_prompt.assert_not_called()
    
    mock_api_instance.repo_info.assert_called_once_with(repo_id=repo_name, repo_type="dataset", token=mock_token)
    mock_create_repo.assert_not_called()

    mock_dataset.save_to_disk.assert_called_once_with(mock_temp_dir_dataset_path)

    mock_git_clone.assert_called_once_with(
        f"https://huggingface.co/datasets/{original_repo}",
        original_os_path_join(mock_temp_dir_clone_path, "original_repo")
    )

    mock_os_walk.assert_called_once_with(original_os_path_join(mock_temp_dir_clone_path, "original_repo"))

    readme_new_path = original_os_path_join(mock_temp_dir_dataset_path, "README.md")
    mock_os_makedirs.assert_any_call(os.path.dirname(readme_new_path), exist_ok=True)
    mock_shutil_copy.assert_any_call(readme_file_original_path, readme_new_path)
    
    card_new_path = original_os_path_join(mock_temp_dir_dataset_path, "subfolder", "dataset_card.md")
    mock_os_makedirs.assert_any_call(os.path.dirname(card_new_path), exist_ok=True)
    mock_shutil_copy.assert_any_call(card_file_original_path, card_new_path)
    
    copied_files = [c_args[0][0] for c_args in mock_shutil_copy.call_args_list]
    assert data_file_original_path not in copied_files
    assert len(copied_files) == 2

    mock_api_instance.upload_folder.assert_called_once_with(
        folder_path=mock_temp_dir_dataset_path,
        repo_id=repo_name,
        repo_type="dataset",
        token=mock_token
    )
    
    its_thorn.postprocessing.console.print.assert_any_call(f"Repository {repo_name} found.")
    its_thorn.postprocessing.console.print.assert_any_call(f"Non-data files from {original_repo} have been copied to the new repository.")
    its_thorn.postprocessing.console.print.assert_any_call(f"Dataset uploaded to https://huggingface.co/datasets/{repo_name}")

# --- Tests for the main 'postprocess' function ---

@patch('its_thorn.postprocessing.inquirer.prompt')
@patch('its_thorn.postprocessing.save_dataset')
@patch('its_thorn.postprocessing.upload_to_hub')
def test_postprocess_save_only_paths_provided(
    mock_upload, mock_save, mock_inquirer_prompt, mock_dataset
):
    output_path = "/provided/path/to/save"
    
    # Simulate user choosing only "Save locally" even if hub_repo could be prompted
    mock_inquirer_prompt.return_value = {"actions": ["Save locally"]}

    postprocess(mock_dataset, output_path=output_path, hub_repo=None)

    mock_inquirer_prompt.assert_called_once() # Should ask for actions
    mock_save.assert_called_once_with(mock_dataset, output_path)
    mock_upload.assert_not_called()
    its_thorn.postprocessing.console.print.assert_any_call("[green]Postprocessing completed successfully![/green]")

@patch('its_thorn.postprocessing.inquirer.prompt')
@patch('its_thorn.postprocessing.save_dataset')
@patch('its_thorn.postprocessing.upload_to_hub')
def test_postprocess_upload_only_paths_provided(
    mock_upload, mock_save, mock_inquirer_prompt, mock_dataset
):
    hub_repo = "user/provided_repo"
    original_repo = "source/dataset"
    
    mock_inquirer_prompt.return_value = {"actions": ["Upload to Hub"]}

    postprocess(mock_dataset, output_path=None, hub_repo=hub_repo, token="test_token", original_repo=original_repo)

    mock_inquirer_prompt.assert_called_once()
    mock_save.assert_not_called()
    mock_upload.assert_called_once_with(mock_dataset, hub_repo, "test_token", original_repo)
    its_thorn.postprocessing.console.print.assert_any_call("[green]Postprocessing completed successfully![/green]")

@patch('its_thorn.postprocessing.inquirer.prompt')
@patch('its_thorn.postprocessing.save_dataset')
@patch('its_thorn.postprocessing.upload_to_hub')
def test_postprocess_save_and_upload_prompt_all(
    mock_upload, mock_save, mock_inquirer_prompt, mock_dataset
):
    # Simulate user choosing both, and needing to be prompted for paths
    prompted_output_path = "/prompted/path"
    prompted_hub_repo = "user/prompted_repo"
    original_repo = "source/dataset_for_upload"

    # Multiple inquirer calls: 1. actions, 2. save path, 3. hub repo name
    mock_inquirer_prompt.side_effect = [
        {"actions": ["Save locally", "Upload to Hub"]},
        {"path": prompted_output_path},
        {"hub": prompted_hub_repo}
    ]
    
    # Note: Token and original_repo are passed for upload part
    postprocess(mock_dataset, output_path=None, hub_repo=None, token="tok", original_repo=original_repo)

    assert mock_inquirer_prompt.call_count == 3
    mock_inquirer_prompt.assert_has_calls([
        call([{'message': 'What actions to perform?', 'name': 'actions', 'type': 'checkbox', 'choices': ['Save locally', 'Upload to Hub'], 'default': ['Save locally']}]),
        call([{'type': 'path', 'name': 'path', 'message': 'Enter the local path to save the dataset:'}]),
        call([{'type': 'text', 'name': 'hub', 'message': 'Enter the name of the HuggingFace Hub repository:'}])
    ])
    
    mock_save.assert_called_once_with(mock_dataset, prompted_output_path)
    mock_upload.assert_called_once_with(mock_dataset, prompted_hub_repo, "tok", original_repo)
    its_thorn.postprocessing.console.print.assert_any_call("[green]Postprocessing completed successfully![/green]")


@patch('its_thorn.postprocessing.inquirer.prompt')
@patch('its_thorn.postprocessing.save_dataset')
@patch('its_thorn.postprocessing.upload_to_hub')
def test_postprocess_no_action_selected(
    mock_upload, mock_save, mock_inquirer_prompt, mock_dataset
):
    mock_inquirer_prompt.return_value = {"actions": []} # User selects no actions

    postprocess(mock_dataset)

    mock_inquirer_prompt.assert_called_once()
    mock_save.assert_not_called()
    mock_upload.assert_not_called()
    its_thorn.postprocessing.console.print.assert_called_with("[yellow]No actions selected. The dataset was not saved or uploaded.[/yellow]")

@patch('its_thorn.postprocessing.inquirer.prompt')
@patch('its_thorn.postprocessing.save_dataset', side_effect=Exception("Save failed!"))
@patch('its_thorn.postprocessing.upload_to_hub')
def test_postprocess_save_fails(
    mock_upload, mock_save, mock_inquirer_prompt, mock_dataset
):
    output_path = "/path/to/save"
    mock_inquirer_prompt.return_value = {"actions": ["Save locally"]}

    with pytest.raises(Exception, match="Save failed!"):
        postprocess(mock_dataset, output_path=output_path)
    
    mock_save.assert_called_once_with(mock_dataset, output_path)
    mock_upload.assert_not_called()
    its_thorn.postprocessing.console.print.assert_any_call(f"[red]Error saving dataset locally: Save failed![/red]")


@patch('its_thorn.postprocessing.inquirer.prompt')
@patch('its_thorn.postprocessing.save_dataset')
@patch('its_thorn.postprocessing.upload_to_hub', side_effect=Exception("Upload failed!"))
def test_postprocess_upload_fails(
    mock_upload, mock_save, mock_inquirer_prompt, mock_dataset
):
    hub_repo = "user/repo"
    mock_inquirer_prompt.return_value = {"actions": ["Upload to Hub"]}

    with pytest.raises(Exception, match="Upload failed!"):
        postprocess(mock_dataset, hub_repo=hub_repo, token="tok", original_repo="orig/repo")
            
    mock_save.assert_not_called()
    mock_upload.assert_called_once_with(mock_dataset, hub_repo, "tok", "orig/repo")
    its_thorn.postprocessing.console.print.assert_any_call(f"[red]Error uploading dataset to HuggingFace Hub: Upload failed![/red]") 