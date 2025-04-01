# eval_best_model.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
# No need to import weights specifically if loading state_dict with weights=None
# from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, Dataset # Import Dataset base class
import os
import numpy as np
import pandas as pd # Needed for creating submission df

# Import evaluation scripts
import eval_cifar100
import eval_ood

# --- Helper Classes (If needed by eval scripts, otherwise optional here) ---
# If eval_ood.py or eval_cifar100.py rely on TransformedSubset,
# you might need to include it here as well, or ensure it's defined
# within those modules appropriately. Let's assume for now they don't
# directly need it, as we'll use standard datasets/loaders here.


def main():
    # --- CONFIGURATION (Match relevant parts of training config) ---
    CONFIG = {
        "model_name": "ResNet50_Pretrained_Fixed", # Informational, matches training
        "batch_size": 32,          # For DataLoader during evaluation
        "dropout_p": 0.5,         # *** CRUCIAL: Must match training config ***
        "num_workers": 0, # Use 0 or match training (4). 0 is often simpler for eval.
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",      # Directory for CIFAR-100
        "ood_dir": "./data/ood-test", # Directory for OOD data
        "best_model_path": "best_model_part2_fixed.pth", # Path to the saved model
        # Add any other config keys needed by eval_ood.evaluate_ood_test
    }
    print("\n--- Evaluation Configuration ---")
    for key, val in CONFIG.items(): print(f"{key}: {val}")
    print("----------------------------\n")

    device = CONFIG["device"]

    # --- Data Transformations (Must match test/val transform from training) ---
    # *** USE IMAGENET STATS (As used in the training script) ***
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Test transform for CIFAR-100 (expects PIL input from CIFAR100 dataset)
    transform_test_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    # OOD Transform (as defined in the training script for OOD evaluation)
    # Assumes OOD dataset loader (inside eval_ood.py) provides Tensors
    # needing ImageNet normalization.
    transform_ood_for_eval = transforms.Compose([
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])
    print("Defined data transforms using ImageNet stats.")

    # --- Load CIFAR-100 Test Data ---
    print("Loading CIFAR-100 test dataset...")
    try:
        # Use standard CIFAR100 dataset, applying the test transform
        testset_cifar = torchvision.datasets.CIFAR100(
            root=CONFIG["data_dir"],
            train=False,
            download=True,
            transform=transform_test_cifar # Apply the specific CIFAR test transform
        )
        testloader_cifar = DataLoader(
            testset_cifar,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["num_workers"],
            pin_memory=True
        )
        print("CIFAR-100 Test DataLoader created.")
    except Exception as e:
        print(f"Error loading CIFAR-100 test data: {e}")
        return # Exit if data cannot be loaded

    # --- Model Setup (Recreate the exact architecture) ---
    print(f"Loading model architecture ({CONFIG['model_name']})...")
    try:
        # Start with the base ResNet50, weights=None because we load from file
        model = models.resnet50(weights=None) # Don't load pretrained weights here
        in_features = model.fc.in_features
        # Recreate the *exact* same final layer structure as during training
        model.fc = nn.Sequential(
            nn.Dropout(p=CONFIG["dropout_p"]), # Use the dropout value from CONFIG
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 100) # CIFAR-100 has 100 classes
        )
        print("Model architecture created.")
    except KeyError as e:
         print(f"Error creating model architecture: Missing key {e} in CONFIG.")
         print("Ensure 'dropout_p' is defined in the CONFIG dictionary of this script.")
         return
    except Exception as e:
        print(f"Error creating model architecture: {e}")
        return

    # --- Load Best Weights ---
    print(f"Loading best model weights from '{CONFIG['best_model_path']}'...")
    if not os.path.exists(CONFIG['best_model_path']):
        print(f"Error: Model weights file not found at {CONFIG['best_model_path']}")
        return

    try:
        # Load the state dictionary
        state_dict = torch.load(CONFIG['best_model_path'], map_location=device)
        # Load the state dictionary into the model structure
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval() # Set model to evaluation mode (disables dropout, etc.)
        print("Best model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # --- Final Evaluation ---
    print("\n--- Starting Final Evaluation ---")

    # 1. Evaluate on Clean CIFAR-100 Test Set
    print("Evaluating on Clean CIFAR-100 Test Set...")
    try:
        # Pass the model, the specific CIFAR-100 testloader, and device
        predictions_clean, clean_accuracy = eval_cifar100.evaluate_cifar100_test(
            model, testloader_cifar, device
        )
        print(f"Final Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")
    except Exception as e:
        print(f"Error during CIFAR-100 evaluation: {e}")
        clean_accuracy = -1 # Indicate failure

    # 2. Evaluate on OOD Data
    print("\nEvaluating on OOD Set...")
    all_predictions_ood = None # Initialize
    try:
        # Pass the model, the evaluation CONFIG, and the specific OOD transform
        all_predictions_ood = eval_ood.evaluate_ood_test(
            model, CONFIG, transform_ood_for_eval
        )
        # Assuming evaluate_ood_test returns a list or dictionary of predictions
        if all_predictions_ood is not None:
             print("OOD evaluation completed.")
        else:
             print("OOD evaluation function returned None.")

    except Exception as e:
        print(f"Error during OOD evaluation: {e}")
        all_predictions_ood = None # Ensure it's None if evaluation fails

    # 3. Create OOD Submission File
    if all_predictions_ood:
        print("\nCreating OOD submission file...")
        try:
            # Assuming create_ood_df takes the predictions dictionary/list
            submission_df_ood = eval_ood.create_ood_df(all_predictions_ood)

            if submission_df_ood is None or submission_df_ood.empty:
                print("Warning: OOD submission DataFrame is empty or creation failed.")
            else:
                submission_csv_path = "submission_ood_part2_fixed.csv" # Consistent naming
                submission_df_ood.to_csv(submission_csv_path, index=False)
                print(f"'{submission_csv_path}' created successfully.")
        except AttributeError:
             print("Error: Looks like 'eval_ood' module might not have 'create_ood_df' function or it failed.")
        except Exception as e:
            print(f"Error creating or saving OOD submission CSV: {e}")
    else:
        print("Skipping OOD submission file creation due to evaluation errors or no predictions returned.")

    print("\nEvaluation script finished.")

if __name__ == '__main__':
    main()