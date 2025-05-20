"""
Crop Image Classification Pipeline

This script implements a complete pipeline for training, validating and testing
image classification models on crop disease datasets.
"""

# ===== Section: Import Libraries =====
import os
import random
import time
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm  # For progress bars

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as TF

# scikit-learn for evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# For visualization
import seaborn as sns
from PIL import Image


# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()


# ===== Section: Configuration Settings =====
# Configuration class to hold all settings
class Config:
    # Paths
    data_dir = "./dataset"
    output_dir = "./models"
    results_dir = "./results"

    # Dataset settings
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # Model settings
    model_type = "resnet50"  # Options: 'custom_cnn', 'resnet18', 'resnet50'
    pretrained = True

    # Training settings
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    weight_decay = 1e-4

    # Early stopping settings
    patience = 5

    # Image settings
    img_size = 224

    # Device settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Create output directories
os.makedirs(Config.output_dir, exist_ok=True)
os.makedirs(Config.results_dir, exist_ok=True)


# ===== Section: Data Preparation =====
def get_data_transforms():
    """Define data transformations for training and validation/testing."""
    # Data augmentation and normalization for training
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(Config.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Just normalization for validation & testing
    val_test_transform = transforms.Compose(
        [
            transforms.Resize(Config.img_size + 32),
            transforms.CenterCrop(Config.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_test_transform


def load_and_split_dataset():
    """Load the dataset and split it into train, validation and test sets."""
    train_transform, val_test_transform = get_data_transforms()

    # Load the full dataset with training transformations
    full_dataset = datasets.ImageFolder(root=Config.data_dir, transform=train_transform)

    # Create a dataset with validation/test transformations
    val_test_dataset = datasets.ImageFolder(
        root=Config.data_dir, transform=val_test_transform
    )

    # Get class names and count
    class_names = full_dataset.classes
    num_classes = len(class_names)

    print(f"Found {num_classes} classes: {class_names}")

    # Calculate splits
    dataset_size = len(full_dataset)
    train_size = int(Config.train_ratio * dataset_size)
    val_size = int(Config.val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    print(f"Total dataset size: {dataset_size}")
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    print(f"Testing set size: {test_size}")

    # Create the splits
    train_dataset, val_dataset_with_aug, test_dataset_with_aug = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Create validation and test datasets with proper transforms
    # We want to keep the same data points but use different transforms
    _, val_dataset_proper, test_dataset_proper = random_split(
        val_test_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    return (
        train_dataset,
        val_dataset_proper,
        test_dataset_proper,
        class_names,
        num_classes,
    )


def create_dataloaders(train_dataset, val_dataset, test_dataset):
    """Create DataLoader objects for train, validation, and test datasets."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ===== Section: Model Architectures =====
class CustomCNN(nn.Module):
    """Custom CNN architecture for baseline comparison."""

    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)


def get_model(model_type, num_classes, pretrained=True):
    """Create a model based on the specified type."""
    print(f"Creating {model_type} model...")

    if model_type == "custom_cnn":
        model = CustomCNN(num_classes)

    elif model_type == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        # Replace the final fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif model_type == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        # Replace the final fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    else:
        raise ValueError(f"Model type {model_type} not supported")

    return model


# ===== Section: Training Functions =====
def train_model(
    model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25
):
    """Train the model with validation and early stopping."""
    since = time.time()

    # Initialize tracking variables
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_epoch = -1

    # For tracking metrics
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    device = Config.device
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params:,} parameters, {trainable_params:,} are trainable")

    # Early stopping setup
    patience = Config.patience
    early_stop_counter = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass - track gradients only in training phase
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calculate metrics for this epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # Store history
            if phase == "train":
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
                if scheduler:
                    scheduler.step()
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # If we got a better validation accuracy, save the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()
                best_epoch = epoch
                early_stop_counter = 0

                # Save the best model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "accuracy": best_acc,
                    },
                    os.path.join(Config.output_dir, f"{Config.model_type}_best.pth"),
                )

            elif phase == "val":
                early_stop_counter += 1

        # Check for early stopping
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs!")
            break

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f} at epoch {best_epoch + 1}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


# ===== Section: Evaluation Functions =====
def evaluate_model(model, test_loader, class_names):
    """Evaluate the model on the test set and return metrics."""
    model.eval()
    model = model.to(Config.device)

    # Lists to store predictions and true labels
    all_preds = []
    all_labels = []

    # No gradient calculation needed for evaluation
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(Config.device)
            labels = labels.to(Config.device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=3
    )

    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    return accuracy, conf_matrix, report


# ===== Section: Visualization Functions =====
def plot_training_history(history, save_path=None):
    """Plot training and validation loss/accuracy curves."""
    plt.figure(figsize=(12, 5))

    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Training Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Training history plot saved to {save_path}")

    plt.show()


def plot_confusion_matrix(conf_matrix, class_names, save_path=None):
    """Plot the confusion matrix as a heatmap."""
    plt.figure(figsize=(12, 10))

    # Normalize the confusion matrix
    norm_conf_matrix = (
        conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
    )

    sns.heatmap(
        norm_conf_matrix,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Normalized Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


# ===== Section: Prediction Demo =====
def predict_random_images(model, test_dataset, class_names, num_images=5):
    """Display and predict random images from the test set."""
    model.eval()
    model = model.to(Config.device)

    # Get a batch of random indices
    indices = torch.randperm(len(test_dataset))[:num_images]

    # Create a figure
    plt.figure(figsize=(15, 3 * num_images))

    for i, idx in enumerate(indices):
        # Get image and label
        image, label = test_dataset[idx]

        # Convert image for display
        image_for_display = image.clone()

        # Make prediction
        image = image.unsqueeze(0).to(Config.device)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        # Get predicted and true class names
        predicted_class = class_names[predicted.item()]
        true_class = class_names[label]

        # Display the image
        plt.subplot(num_images, 1, i + 1)
        plt.imshow(np.transpose(image_for_display.cpu().numpy(), (1, 2, 0)))

        # Normalize the image for better display
        plt.title(f"True: {true_class} | Predicted: {predicted_class}")
        plt.axis("off")

        # Color based on correctness
        if predicted.item() == label:
            plt.title(
                f"True: {true_class} | Predicted: {predicted_class}", color="green"
            )
        else:
            plt.title(f"True: {true_class} | Predicted: {predicted_class}", color="red")

    plt.tight_layout()
    plt.savefig(os.path.join(Config.results_dir, "sample_predictions.png"))
    plt.show()


# ===== Section: Main Execution =====
def main():
    """Main execution function"""
    # 1. Load and prepare data
    print("Preparing datasets...")
    train_dataset, val_dataset, test_dataset, class_names, num_classes = (
        load_and_split_dataset()
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    dataloaders = {"train": train_loader, "val": val_loader}

    # 2. Create model
    model = get_model(Config.model_type, num_classes, Config.pretrained)
    model = model.to(Config.device)

    # 3. Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Different learning rates for different parts of the model if using transfer learning
    if Config.model_type != "custom_cnn" and Config.pretrained:
        # Parameters of newly constructed modules have requires_grad=True by default
        params_to_update = []
        params_to_fine_tune = []

        # Split parameters into those to update and those to fine-tune
        for name, param in model.named_parameters():
            if "fc" in name:  # Parameters in the final classifier layer
                params_to_update.append(param)
            else:
                params_to_fine_tune.append(param)

        # Optimizer with different learning rates
        optimizer = optim.Adam(
            [
                {"params": params_to_fine_tune, "lr": Config.learning_rate / 10},
                {"params": params_to_update},
            ],
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        )
    else:
        # Use the same learning rate for all parameters in custom CNN
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 4. Train the model
    print(f"\nTraining {Config.model_type} model on {Config.device}...")
    model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler, Config.num_epochs
    )

    # 5. Evaluate on test set
    print("\nEvaluating on test set...")
    accuracy, conf_matrix, report = evaluate_model(model, test_loader, class_names)

    # 6. Visualize results
    print("\nCreating visualizations...")

    # Plot training history
    plot_training_history(
        history,
        save_path=os.path.join(
            Config.results_dir, f"{Config.model_type}_training_history.png"
        ),
    )

    # Plot confusion matrix
    plot_confusion_matrix(
        conf_matrix,
        class_names,
        save_path=os.path.join(
            Config.results_dir, f"{Config.model_type}_confusion_matrix.png"
        ),
    )

    # 7. Demo predictions on random images
    print("\nGenerating sample predictions...")
    predict_random_images(model, test_dataset, class_names, num_images=5)

    # 8. Save final model
    torch.save(
        model.state_dict(),
        os.path.join(Config.output_dir, f"{Config.model_type}_final.pth"),
    )
    print(
        f"Final model saved to {os.path.join(Config.output_dir, f'{Config.model_type}_final.pth')}"
    )

    # 9. Save model configuration and results
    report_str = report.replace("\n", "<br>")
    with open(os.path.join(Config.results_dir, "model_report.md"), "w") as f:
        f.write(f"# Model Training Report\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- Model: {Config.model_type}\n")
        f.write(f"- Pretrained: {Config.pretrained}\n")
        f.write(f"- Image size: {Config.img_size}x{Config.img_size}\n")
        f.write(f"- Batch size: {Config.batch_size}\n")
        f.write(f"- Learning rate: {Config.learning_rate}\n")
        f.write(f"- Weight decay: {Config.weight_decay}\n\n")
        f.write(f"## Results\n\n")
        f.write(f"- Best test accuracy: {accuracy:.4f}\n\n")
        f.write(f"## Classification Report\n\n")
        f.write(f"```\n{report}\n```\n")

    print(
        f"Model report saved to {os.path.join(Config.results_dir, 'model_report.md')}"
    )

    # 10. Save training history as JSON for future reference
    history_dict = {
        "train_loss": [float(val) for val in history["train_loss"]],
        "val_loss": [float(val) for val in history["val_loss"]],
        "train_acc": [float(val) for val in history["train_acc"]],
        "val_acc": [float(val) for val in history["val_acc"]],
    }

    with open(os.path.join(Config.results_dir, "training_history.json"), "w") as f:
        json.dump(history_dict, f, indent=4)

    print(
        f"Training history saved to {os.path.join(Config.results_dir, 'training_history.json')}"
    )


if __name__ == "__main__":
    main()
