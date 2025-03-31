# part2.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import random_split
from tqdm.auto import tqdm
import wandb
import os

def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    """Train one epoch."""
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})
    
    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def validate(model, valloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})
    
    val_loss = running_loss / len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def main():
    ############################################################################
    #    Configuration Dictionary
    ############################################################################
    CONFIG = {
        "model": "ResNet18_Pretrained",
        "batch_size": 32,
        "learning_rate": 0.01,
        "epochs": 50,  # Increase the maximum epochs
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge-part2",
        "seed": 42,
    }

    print("\nCONFIG Dictionary:")
    for key, val in CONFIG.items():
        print(f"{key}: {val}")

    ############################################################################
    #      Data Transformations
    ############################################################################
    # Training transforms include some augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Validation/Test transforms should not include augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    ############################################################################
    #       Data Loading (CIFAR-100)
    ############################################################################
    trainset_full = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True,
                                                  download=True, transform=transform_train)
    train_size = int(0.8 * len(trainset_full))
    val_size = len(trainset_full) - train_size
    trainset, valset = random_split(trainset_full, [train_size, val_size])
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"],
                                              shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"],
                                            shuffle=False, num_workers=CONFIG["num_workers"])

    testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False,
                                             download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"],
                                             shuffle=False, num_workers=CONFIG["num_workers"])

    ############################################################################
    #   Instantiate and Prepare the Pretrained Model
    ############################################################################
    # Load a pretrained ResNet-18 model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(CONFIG["device"])

    print("\nModel summary:")
    print(model)

    ############################################################################
    # Loss Function, Optimizer, and Learning Rate Scheduler
    ############################################################################
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG["learning_rate"], momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    ############################################################################
    # Initialize Weights & Biases
    ############################################################################
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    wandb.watch(model)

    ############################################################################
    # --- Training Loop ---
    ############################################################################
    best_val_acc = 0.0
    patience = 5         # Number of epochs to wait for an improvement before stopping
    epochs_no_improve = 0

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        # Check if the validation accuracy has improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_part2.pth")
            wandb.save("best_model_part2.pth")
            epochs_no_improve = 0  # reset counter on improvement
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        # Early stopping condition
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    wandb.finish()

    ############################################################################
    # Evaluation on the Test Set
    ############################################################################
    test_loss, test_acc = validate(model, testloader, criterion, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {test_acc:.2f}%")

    ############################################################################
    # Evaluation -- shouldn't have to change the following code
    ############################################################################
    import eval_cifar100
    import eval_ood

    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD ---
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # --- Create Submission File (OOD) ---
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood_part2.csv", index=False)
    print("submission_ood_part2.csv created successfully.")

if __name__ == '__main__':
    main()
