# eval_ood.py - CORRECTED

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F # Import functional API
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import wandb
import urllib.request

# Custom Dataset for OOD evaluation
class OODImageDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for OOD images.
    Loads numpy array from a specific severity subset, converts to tensor [0, 1],
    permutes dims, and applies a specified transform.
    """
    def __init__(self, images_npy_path, severity, transform=None):
        """
        Args:
            images_npy_path (str): Path to the .npy file for a distortion.
            severity (int): Severity level (1-5).
            transform (callable, optional): Optional transform to be applied
                                            on a tensor image (C x H x W).
        """
        try:
            # Load the full distortion data
            all_images = np.load(images_npy_path)
        except FileNotFoundError:
            print(f"Error: NPY file not found at {images_npy_path}")
            # Handle error: raise exception or create an empty dataset
            raise FileNotFoundError(f"Required NPY file missing: {images_npy_path}")
            # Alternatively: self.images_tensor = torch.empty((0, 3, 32, 32))

        # Select the subset for the given severity
        start_index = (severity - 1) * 10000
        end_index = severity * 10000

        # Basic validation for indices
        if start_index >= len(all_images) or end_index > len(all_images) or start_index < 0:
             raise IndexError(f"Severity {severity} leads to invalid indices [{start_index}:{end_index}] for file {images_npy_path} with length {len(all_images)}")

        images_subset = all_images[start_index:end_index]

        # Convert to PyTorch tensors, normalize to [0, 1], and permute
        # (N, H, W, C) -> (N, C, H, W)
        self.images_tensor = torch.from_numpy(images_subset).float() / 255.0
        self.images_tensor = self.images_tensor.permute(0, 3, 1, 2)

        self.transform = transform

    def __len__(self):
        return len(self.images_tensor)

    def __getitem__(self, idx):
        # Get the tensor image (already C x H x W, [0, 1])
        image = self.images_tensor[idx]

        # Apply the transform if it exists
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error applying transform to image index {idx}. Error: {e}")
                # Depending on severity, might return original image or raise error
                # Returning original might lead to size mismatches later. Best to debug the transform.
                # For now, let's re-raise to make the issue visible.
                raise e

        return image # Return only the transformed image

# --- Evaluation on OOD Data ---
def evaluate_ood(model, distortion_name, severity, CONFIG, transform_ood):
    """
    Evaluates the model on a specific OOD distortion and severity.

    Args:
        model: The trained PyTorch model.
        distortion_name (str): Name of the distortion (e.g., 'distortion00').
        severity (int): Severity level (1-5).
        CONFIG (dict): Configuration dictionary.
        transform_ood (callable): The torchvision transform pipeline to apply
                                   to the OOD images (should match test transform).
    Returns:
        list: List of predicted labels for the OOD subset.
    """
    ood_dir = CONFIG["ood_dir"]
    device = CONFIG["device"]
    npy_path = os.path.join(ood_dir, f"{distortion_name}.npy")

    try:
        # Create the dataset for this specific distortion/severity
        dataset = OODImageDataset(npy_path, severity, transform=transform_ood)
    except FileNotFoundError:
        # If file not found even after download check, return empty list for this batch
        return []
    except IndexError as e:
        print(f"Error creating dataset for {distortion_name} severity {severity}: {e}")
        return [] # Skip this severity if indexing is wrong

    # Check if dataset is empty (e.g., if file loading failed gracefully inside Dataset)
    if len(dataset) == 0:
        print(f"Warning: Empty dataset for {distortion_name} severity {severity}.")
        return []

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True) # Set pin_memory=True if using GPU

    predictions = []
    model.eval() # Ensure model is in evaluation mode
    with torch.no_grad():
        # Dataloader now yields pre-transformed batches
        for inputs in tqdm(dataloader, desc=f"Evaluating {distortion_name} (Severity {severity})", leave=False):
            try:
                inputs = inputs.to(device, non_blocking=True) # Use non_blocking with pin_memory=True
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                predictions.extend(predicted.cpu().numpy())
            except Exception as e:
                print(f"Error during model inference on batch for {distortion_name} severity {severity}. Error: {e}")
                # Handle error, e.g., skip batch, add dummy predictions?
                # Adding dummy predictions might skew results silently. Best to debug.
                # For now, just print error and continue, leading to fewer predictions.
                continue # Skip this batch

    return predictions


# Check if the files are already downloaded
def files_already_downloaded(directory, num_files):
    """Checks if all expected distortion NPY files exist."""
    if not os.path.isdir(directory):
        return False
    for i in range(num_files):
        file_name = f"distortion{i:02d}.npy"
        file_path = os.path.join(directory, file_name)
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}") # Log which file is missing
            return False
    return True

# Download OOD files if needed
def download_ood_files(CONFIG):
    """Downloads OOD dataset files if they don't exist."""
    ood_dir = CONFIG["ood_dir"]
    # Ensure ood_dir exists in CONFIG, provide default if not
    if not ood_dir:
        ood_dir = "./data/ood-test" # Default location
        print(f"Warning: 'ood_dir' not in CONFIG, using default: {ood_dir}")

    num_files = 19  # Number of files to download (distortion00 to distortion18)

    if not files_already_downloaded(ood_dir, num_files):
        print("OOD files not found or incomplete. Downloading...")
        os.makedirs(ood_dir, exist_ok=True)
        base_url = "https://github.com/DL4DS/ood-test-files/raw/refs/heads/main/ood-test/"

        download_success = True
        for i in range(num_files):
            file_name = f"distortion{i:02d}.npy"
            file_url = base_url + file_name
            file_path = os.path.join(ood_dir, file_name)

            # Download only if the specific file is missing
            if not os.path.exists(file_path):
                print(f"Downloading {file_name}...")
                try:
                    urllib.request.urlretrieve(file_url, file_path)
                    print(f"Downloaded {file_name} to {file_path}")
                except Exception as e:
                    print(f"Error downloading {file_name}: {e}")
                    print("Please check your internet connection or the URL.")
                    # Decide how to handle failed download (e.g., raise error, return False)
                    download_success = False # Indicate download failed for at least one file
            # else:
                 # print(f"{file_name} already exists.") # Reduce verbosity

        if download_success:
            print("OOD file download check complete.")
        else:
            print("Warning: Some OOD files failed to download.")
        return download_success # Return status
    else:
        print("All OOD files are already downloaded.")
        return True # Indicate files already present

def evaluate_ood_test(model, CONFIG, transform_ood):
    """
    Evaluates the model on the full OOD test set (all distortions and severities).

    Args:
        model: The trained PyTorch model.
        CONFIG (dict): Configuration dictionary containing 'ood_dir', 'batch_size', etc.
        transform_ood (callable): The torchvision transform pipeline to apply
                                   to the OOD images (should match test transform).
    Returns:
        list: List of all predictions concatenated for the submission file.
              Returns an empty list if evaluation cannot proceed.
    """
    # Ensure OOD files are present before proceeding
    if not download_ood_files(CONFIG):
        print("Cannot proceed with OOD evaluation due to download issues.")
        return [] # Return empty list

    ood_dir = CONFIG.get("ood_dir", "./data/ood-test") # Get ood_dir safely

    distortions = [f"distortion{str(i).zfill(2)}" for i in range(19)]
    all_predictions = []
    processed_distortions_count = 0

    model.eval() # Ensure model is in evaluation mode
    print("\nStarting OOD Evaluation...")
    overall_progress = tqdm(total=19 * 5, desc="Overall OOD Progress") # Outer progress bar

    for distortion in distortions:
        npy_path = os.path.join(ood_dir, f"{distortion}.npy")
        if not os.path.exists(npy_path):
             print(f"Skipping {distortion}: File not found at {npy_path}")
             overall_progress.update(5) # Update progress for the 5 skipped severities
             continue # Skip this distortion if file missing

        dist_predictions_count = 0
        for severity in range(1, 6):
            # Pass the transform_ood down to the evaluation function
            predictions = evaluate_ood(model, distortion, severity, CONFIG, transform_ood)
            all_predictions.extend(predictions)
            dist_predictions_count += len(predictions)
            overall_progress.update(1) # Update progress after each severity

        # Simple check if predictions were generated for this distortion
        if dist_predictions_count > 0:
            processed_distortions_count += 1

    overall_progress.close()
    print(f"Finished OOD Evaluation. Processed {processed_distortions_count}/19 distortions.")
    return all_predictions


def create_ood_df(all_predictions):
    """Creates the OOD submission DataFrame."""
    if not all_predictions: # Handle case where evaluation failed or yielded no results
        print("Error: No predictions were generated for OOD submission.")
        return pd.DataFrame({'id': [], 'label': []})

    distortions = [f"distortion{str(i).zfill(2)}" for i in range(19)]
    expected_total_predictions = 19 * 5 * 10000 # 950,000

    # --- Create Submission File (OOD) ---
    # Create IDs for OOD ONLY for the distortions that were actually processed
    # This requires knowing which NPY files exist. Re-check existence.
    ids_ood = []
    ood_dir = "./data/ood-test" # Assuming default location - might need to get from CONFIG if it changes
    processed_distortion_names = [] # Keep track of distortions we generate IDs for

    print("Generating IDs for submission...")
    for distortion in distortions:
        npy_path = os.path.join(ood_dir, f"{distortion}.npy")
        if os.path.exists(npy_path):
            processed_distortion_names.append(distortion)
            for severity in range(1, 6):
                for i in range(10000):
                    ids_ood.append(f"{distortion}_{severity}_{i}")
        else:
            print(f"Note: Skipping ID generation for missing file {distortion}.npy")

    expected_ids_count = len(processed_distortion_names) * 5 * 10000

    # Crucial Check: Compare expected IDs based on existing files vs actual predictions
    if expected_ids_count != len(all_predictions):
         print(f"\n!!! CRITICAL WARNING !!!")
         print(f"Mismatch between expected IDs based on existing files ({expected_ids_count}) and actual predictions generated ({len(all_predictions)}).")
         print("This likely indicates an error during the evaluation of one or more distortion/severity levels.")
         print("The generated submission file will likely be INVALID for Kaggle.")
         print("Please check the evaluation logs for errors.")
         # Decide how to proceed: return empty, raise error, or try to create partial?
         # Returning partial is risky for Kaggle. Best to signal failure clearly.
         return pd.DataFrame({'id': [], 'label': []}) # Return empty DataFrame to indicate failure

    print(f"Generated {len(ids_ood)} IDs, matching {len(all_predictions)} predictions.")
    submission_df_ood = pd.DataFrame({'id': ids_ood, 'label': all_predictions})
    return submission_df_ood

# ===========================================================
# REMINDER: The specific transform_ood object
# (e.g., for ResNet18 or EfficientNet) MUST be defined in and
# passed from the calling script (part2.py, part3.py, eval_best_model.py).
# ===========================================================