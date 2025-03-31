# eval_best_model_part2.py

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import eval_cifar100
import eval_ood

def main():
    # Configuration dictionary (update as needed)
    CONFIG = {
        "batch_size": 32,
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    device = CONFIG["device"]

    # Instantiate the model architecture (must match what was used during training)
    # Load a pretrained EfficientNet_B0 model and modify its classifier
    model = models.efficientnet_b0(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 100)  # CIFAR-100 has 100 classes
    model = model.to(device)

    # Load the best saved model weights
    checkpoint_path = "best_model_part3.pth"
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded best model from '{checkpoint_path}'")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()  # Set model to evaluation mode

    # Define test transforms (should match those used during evaluation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-100 test dataset
    testset = torchvision.datasets.CIFAR100(
        root=CONFIG["data_dir"], 
        train=False, 
        download=True, 
        transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        num_workers=CONFIG["num_workers"]
    )

    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, device)
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD Data ---
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_csv_path = "submission_ood_part2_evaluator.csv"
    submission_df_ood.to_csv(submission_csv_path, index=False)
    print(f"{submission_csv_path} created successfully.")

if __name__ == '__main__':
    main()
