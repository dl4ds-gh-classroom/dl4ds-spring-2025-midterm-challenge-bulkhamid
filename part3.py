# part3.py - UPDATED

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models # Keep this
from torchvision.models import EfficientNet_B0_Weights # Import weights enum
from torch.utils.data import random_split
from tqdm import tqdm # Use auto submodule for better notebook compatibility if needed
import wandb
import os # Needed for os.path.exists checks potentially

# Import evaluation scripts
import eval_cifar100
import eval_ood

def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        # Corrected formatting:
        progress_bar.set_postfix({"loss": f"{running_loss/(i+1):.4f}", "acc": f"{100.*correct/total:.2f}%"})
    return running_loss/len(trainloader), 100.*correct/total

def validate(model, valloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            # Corrected formatting:
            progress_bar.set_postfix({"loss": f"{running_loss/(i+1):.4f}", "acc": f"{100.*correct/total:.2f}%"})
    return running_loss/len(valloader), 100.*correct/total

def main():
    CONFIG = {
        "model": "EfficientNet_B0_opt", # Your updated model name
        "batch_size": 32,
        "learning_rate": 1e-4,  # Your lowered LR
        "epochs": 100,          # Your increased epochs
        "patience": 10,         # Increased patience for early stopping
        "weight_decay": 1e-4,   # Your weight decay
        "label_smoothing": 0.1, # Your label smoothing factor
        "num_workers": 4,       # Adjust based on your system
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",
        "ood_dir": "./data/ood-test", # Ensure this directory exists
        "wandb_project": "sp25-ds542-challenge-part3", # Your WandB project
        "seed": 42,
        "best_model_path": "best_model_part3.pth", # Define path for saving best model
    }

    print("\n--- Configuration ---")
    for key, val in CONFIG.items():
        print(f"{key}: {val}")
    print("---------------------\n")

    # For reproducibility
    torch.manual_seed(CONFIG["seed"])
    if CONFIG["device"] == "cuda":
        torch.cuda.manual_seed(CONFIG["seed"])
        # Optional: CUDNN settings for reproducibility (can slow down training)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    device = CONFIG["device"] # Cache device

    # --- Data Transformations ---
    # Using ImageNet stats as EfficientNet was pretrained on it
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # 1. Training Transforms (with augmentations)
    transform_train = transforms.Compose([
        transforms.Resize(256), # EfficientNet often works better with larger images
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # Added ColorJitter
        transforms.RandAugment(num_ops=2, magnitude=9), # Added RandAugment (adjust ops/magnitude as needed)
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # 2. Validation/Test Transforms (only necessary resizing/cropping and normalization)
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # 3. OOD Transform (MUST match test transform but handle Tensor input)
    transform_ood_for_eval = transforms.Compose([
        transforms.ToPILImage(), # Convert input Tensor (from OODImageDataset) to PIL
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),   # Convert back to Tensor
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    print("Defined data transforms (Train, Test, OOD).")

    # --- Data Loading ---
    print("Loading datasets...")
    try:
        # Load full training set
        trainset_full = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, download=True, transform=transform_train)

        # Split into train/validation
        train_size = int(0.8 * len(trainset_full))
        val_size = len(trainset_full) - train_size
        generator = torch.Generator().manual_seed(CONFIG["seed"]) # Use generator for reproducible split
        trainset, valset = random_split(trainset_full, [train_size, val_size], generator=generator)
        # Apply test transform to validation set (important for fair evaluation)
        # Create a wrapper or re-assign transform (safer to wrap)
        class ApplyTransform(torch.utils.data.Dataset):
            def __init__(self, dataset, transform):
                self.dataset = dataset
                self.transform = transform
            def __getitem__(self, index):
                x, y = self.dataset[index] # Original dataset provides PIL image here
                return self.transform(x), y
            def __len__(self):
                return len(self.dataset)
        # If valset already has transform_train, need to access underlying data or be careful
        # Assuming valset from random_split inherits from trainset_full structure:
        valset.dataset.transform = transform_test # Modify the transform applied by the underlying dataset

        # Load test set
        testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False, download=True, transform=transform_test)

        # Create DataLoaders
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
        print("Datasets and DataLoaders created.")
    except Exception as e:
        print(f"Error during data loading: {e}")
        return

    # --- Model Setup ---
    print("Loading pretrained EfficientNet_B0 model...")
    # Use the weights enum for clarity and future-proofing
    weights = EfficientNet_B0_Weights.DEFAULT # Or EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    # Modify the classifier for CIFAR-100 (100 classes)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 100)
    model = model.to(device)
    print("Model loaded and moved to device.")

    # --- Loss, Optimizer, Scheduler ---
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    # AdamW is often good for fine-tuning Transformers and ConvNets
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    # Cosine annealing is a popular choice for smooth LR decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    print("Loss function, optimizer, and scheduler configured.")

    # --- Weights & Biases Initialization ---
    try:
        wandb.init(project=CONFIG["wandb_project"], config=CONFIG, name=CONFIG["model"]) # Added run name
        wandb.watch(model, log="all", log_freq=100) # Watch gradients and parameters
        print("Weights & Biases initialized.")
    except Exception as e:
        print(f"Error initializing Weights & Biases: {e}. Training will continue without W&B logging.")
        wandb.init(mode="disabled") # Disable wandb if init fails


    # --- Training Loop ---
    print("\n--- Starting Training ---")
    best_val_acc = 0.0
    epochs_no_improve = 0
    patience = CONFIG["patience"]

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, device) # Use same criterion for validation loss

        # Step the scheduler AFTER the optimizer step and validation
        scheduler.step()

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"] # Log learning rate
        })

        print(f"Epoch {epoch+1}/{CONFIG['epochs']}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Check for improvement and save best model
        if val_acc > best_val_acc:
            print(f"Validation accuracy improved ({best_val_acc:.2f}% -> {val_acc:.2f}%). Saving model...")
            best_val_acc = val_acc
            epochs_no_improve = 0  # Reset counter
            try:
                torch.save(model.state_dict(), CONFIG["best_model_path"])
                # wandb.save(CONFIG["best_model_path"]) # Wandb saving can be slow, optional
            except Exception as e:
                print(f"Error saving model: {e}")
        else:
            epochs_no_improve += 1
            print(f"No validation accuracy improvement for {epochs_no_improve} epoch(s).")

        # Early stopping condition
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print("--- Training Finished ---")
    wandb.finish() # Finish W&B run

    # --- Final Evaluation ---
    print("\n--- Starting Final Evaluation ---")
    # Load the best model saved during training for final evaluation
    print(f"Loading best model weights from '{CONFIG['best_model_path']}' for final evaluation...")
    try:
        # Re-instantiate model architecture and load best weights
        # Important if early stopping occurred, otherwise model holds last epoch's weights
        model = models.efficientnet_b0(weights=None) # Load architecture
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 100)
        state_dict = torch.load(CONFIG['best_model_path'], map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval() # Ensure evaluation mode
        print("Best model loaded successfully.")
    except Exception as e:
        print(f"Error loading best model weights from {CONFIG['best_model_path']}: {e}")
        print("Proceeding with the model state from the end of training (which might not be the best).")
        # Ensure model is still in eval mode if error occurred after loading attempt
        model.eval()


    # 1. Evaluate on Clean CIFAR-100 Test Set
    print("Evaluating on Clean Test Set...")
    # Use the separate eval_cifar100 script for consistency if desired,
    # or just use the validate function again with the testloader.
    # Using validate function:
    # test_loss, test_acc = validate(model, testloader, criterion, device)
    # print(f"Final Clean CIFAR-100 Test Accuracy (using validate): {test_acc:.2f}%")

    # Using eval_cifar100 script (assuming it takes model, loader, device):
    predictions_clean, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, device)
    print(f"Final Clean CIFAR-100 Test Accuracy (using eval_cifar100): {clean_accuracy:.2f}%")


    # 2. Evaluate on OOD Data
    print("\nEvaluating on OOD Set...")
    # **PASS THE CORRECT OOD TRANSFORM** defined earlier
    all_predictions_ood = eval_ood.evaluate_ood_test(model, CONFIG, transform_ood_for_eval)


    # 3. Create OOD Submission File
    if all_predictions_ood: # Check if predictions were generated
        print("\nCreating OOD submission file...")
        submission_df_ood = eval_ood.create_ood_df(all_predictions_ood)
        if submission_df_ood.empty:
             print("Failed to create OOD submission DataFrame (likely due to prediction count mismatch).")
        else:
            submission_csv_path = "submission_ood_part3.csv" # Standard name for Part 3
            try:
                submission_df_ood.to_csv(submission_csv_path, index=False)
                print(f"'{submission_csv_path}' created successfully.")
            except Exception as e:
                print(f"Error saving OOD submission CSV: {e}")
    else:
        print("Skipping OOD submission file creation as no OOD predictions were generated.")

    print("\nScript finished.")


if __name__ == '__main__':
    main()