# part2_improved.py - FIXED NORMALIZATION & LR

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet50_Weights # Correct weights import
from torch.utils.data import random_split, Dataset, Subset
from tqdm.auto import tqdm
import wandb
import os
import numpy as np

# Import evaluation scripts
import eval_cifar100
import eval_ood

# --- Helper Classes (Keep TransformedSubset as defined before) ---
class TransformedSubset(Dataset):
    def __init__(self, full_dataset, indices, transform):
        self.full_dataset = full_dataset
        self.indices = indices
        self.transform = transform
        # Check if transform pipeline starts with ToPILImage
        self._needs_to_pil = not (self.transform and self.transform.transforms and isinstance(self.transform.transforms[0], transforms.ToPILImage))

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        img, label = self.full_dataset.data[original_idx], self.full_dataset.targets[original_idx]

        # Convert numpy array to PIL Image if necessary
        if self._needs_to_pil:
             # Ensure img is numpy HWC before converting
             if isinstance(img, np.ndarray) and img.ndim == 3:
                  img = transforms.functional.to_pil_image(img)
             # Add handling if img is already PIL or other format if needed

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.indices)


# --- MixUp Functions (Keep as is) ---
def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- Training/Validation Functions (Keep as is with f-string fix) ---
def train(epoch, model, trainloader, optimizer, criterion, CONFIG, mixup_alpha=0.2):
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    correct = 0; total = 0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=mixup_alpha, device=device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (lam * predicted.eq(targets_a).sum().item() + (1 - lam) * predicted.eq(targets_b).sum().item())
        progress_bar.set_postfix({"loss": f"{running_loss / (i + 1):.4f}", "acc": f"{100. * correct / total:.2f}%"})
    return running_loss / len(trainloader), 100. * correct / total

def validate(model, valloader, criterion, device):
    model.eval(); running_loss = 0.0; correct = 0; total = 0
    progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0); correct += predicted.eq(labels).sum().item()
            progress_bar.set_postfix({"loss": f"{running_loss / (i + 1):.4f}", "acc": f"{100. * correct / total:.2f}%"})
    return running_loss / len(valloader), 100. * correct / total

# --- Main Script ---
def main():
    # --- CONFIGURATION ---
    CONFIG = {
        "model": "ResNet50_Pretrained_Fixed", # Renamed model ID
        "batch_size": 32,
        "learning_rate": 1e-4,  # *** LOWERED LEARNING RATE ***
        "epochs": 100,
        "patience": 10,
        "weight_decay": 1e-4,
        "mixup_alpha": 0.2,
        "dropout_p": 0.5,
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data", "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge-part2-improved", "seed": 42,
        "best_model_path": "best_model_part2_fixed.pth", # New save path
    }
    print("\n--- Configuration ---")
    for key, val in CONFIG.items(): print(f"{key}: {val}")
    print("---------------------\n")

    # --- Reproducibility ---
    torch.manual_seed(CONFIG["seed"])
    if CONFIG["device"] == "cuda": torch.cuda.manual_seed(CONFIG["seed"])
    device = CONFIG["device"]

    # --- Data Transformations ---
    # *** USE IMAGENET STATS ***
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        # Expects PIL
        transforms.RandomCrop(32, padding=4), # Keep CIFAR augmentations
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        # Consider adding RandAugment here?
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std) # *** FIXED ***
    ])

    transform_test = transforms.Compose([
        # Expects PIL
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std) # *** FIXED ***
    ])

    # OOD Transform - expects Tensor input from OODImageDataset
    transform_ood_for_eval = transforms.Compose([
        # Needs ToPILImage FIRST if OODImageDataset gives raw numpy/tensor [0,1]
        # Let's assume eval_ood.py's OODImageDataset gives C,H,W tensor [0,1]
        # But ResNet expects ImageNet normalization.
        # We *cannot* apply ToPILImage here as input is already Tensor.
        # SO: Normalize directly.
        transforms.Normalize(imagenet_mean, imagenet_std) # *** FIXED ***
    ])
    print("Defined data transforms (Train, Test, OOD) using ImageNet stats.")


    # --- Data Loading --- (Using TransformedSubset)
    print("Loading datasets...")
    try:
        trainset_full = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, download=True)
        testset_base = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False, download=True)

        train_size = int(0.8 * len(trainset_full)); val_size = len(trainset_full) - train_size
        generator = torch.Generator().manual_seed(CONFIG["seed"])
        indices = torch.randperm(len(trainset_full), generator=generator).tolist()
        train_indices, val_indices = indices[:train_size], indices[train_size:]

        trainset = TransformedSubset(trainset_full, train_indices, transform_train)
        valset = TransformedSubset(trainset_full, val_indices, transform_test)
        testset = TransformedSubset(testset_base, list(range(len(testset_base))), transform_test)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
        print("Datasets and DataLoaders created.")
    except Exception as e:
        print(f"Error during data loading: {e}"); raise

    # --- Model Setup --- (Keep as is)
    print("Loading pretrained ResNet50 model...")
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=CONFIG["dropout_p"]), nn.BatchNorm1d(in_features), nn.Linear(in_features, 100)
    )
    model = model.to(device)
    print("Model loaded and moved to device.")

    # --- Loss, Optimizer, Scheduler ---
    criterion = nn.CrossEntropyLoss()
    # *** SWITCHED TO AdamW and using new LR ***
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    print("Loss function, AdamW optimizer, and Cosine scheduler configured.")

    # --- WandB Init --- (Keep as is)
    try:
        wandb.init(project=CONFIG["wandb_project"], config=CONFIG, name=CONFIG["model"])
        wandb.watch(model, log="all", log_freq=100)
        print("Weights & Biases initialized.")
    except Exception as e:
        print(f"Error initializing Weights & Biases: {e}. Disabling W&B."); wandb.init(mode="disabled")

    # --- Training Loop --- (Keep as is)
    print("\n--- Starting Training ---"); best_val_acc = 0.0; epochs_no_improve = 0; patience = CONFIG["patience"]
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG, mixup_alpha=CONFIG["mixup_alpha"])
        val_loss, val_acc = validate(model, valloader, criterion, device)
        scheduler.step()
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,"val_loss": val_loss, "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            print(f"Validation accuracy improved ({best_val_acc:.2f}% -> {val_acc:.2f}%). Saving model..."); best_val_acc = val_acc; epochs_no_improve = 0
            try: torch.save(model.state_dict(), CONFIG["best_model_path"])
            except Exception as e: print(f"Error saving model: {e}")
        else:
            epochs_no_improve += 1; print(f"No validation accuracy improvement for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= patience: print(f"Early stopping triggered after {epoch+1} epochs."); break
    print("--- Training Finished ---"); wandb.finish()

    # --- Final Evaluation ---
    print("\n--- Starting Final Evaluation ---")
    print(f"Loading best model weights from '{CONFIG['best_model_path']}'...")
    try: # Reload best model
        model = models.resnet50(weights=None); in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=CONFIG["dropout_p"]), nn.BatchNorm1d(in_features), nn.Linear(in_features, 100))
        state_dict = torch.load(CONFIG['best_model_path'], map_location=device)
        model.load_state_dict(state_dict); model = model.to(device); model.eval()
        print("Best model loaded successfully.")
    except Exception as e:
        print(f"Error loading best model weights: {e}. Evaluating model from last training epoch."); model.eval()

    # 1. Evaluate on Clean CIFAR-100 Test Set
    print("Evaluating on Clean Test Set...")
    predictions_clean, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, device)
    print(f"Final Clean CIFAR-100 Test Accuracy (Eval Script): {clean_accuracy:.2f}%")

    # 2. Evaluate on OOD Data
    print("\nEvaluating on OOD Set...")
    # **PASS THE CORRECT OOD TRANSFORM (which now includes ImageNet norm)**
    all_predictions_ood = eval_ood.evaluate_ood_test(model, CONFIG, transform_ood_for_eval)

    # 3. Create OOD Submission File
    if all_predictions_ood:
        print("\nCreating OOD submission file...")
        submission_df_ood = eval_ood.create_ood_df(all_predictions_ood)
        if submission_df_ood.empty: print("Failed to create OOD submission DataFrame.")
        else:
            submission_csv_path = "submission_ood_part2_fixed.csv" # New name
            try: submission_df_ood.to_csv(submission_csv_path, index=False); print(f"'{submission_csv_path}' created successfully.")
            except Exception as e: print(f"Error saving OOD submission CSV: {e}")
    else: print("Skipping OOD submission file creation.")

    print("\nScript finished.")

if __name__ == '__main__':
    main()