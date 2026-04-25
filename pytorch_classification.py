import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import os
import argparse
import kagglehub
from tqdm import tqdm

# Model Definition
class NNClassifier(nn.Module):
    def __init__(self):
        super(NNClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(67500, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # Outputting 6 raw scores (logits)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits

# Data Preparation
def get_transforms():
    """Get the transforms for training and inference. Must match exactly."""
    return transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_datasets(data_path, transform):
    """Load train and test datasets."""
    path_train = os.path.join(data_path, "seg_train", "seg_train")
    path_test = os.path.join(data_path, "seg_test", "seg_test")

    train_dataset = datasets.ImageFolder(root=path_train, transform=transform)
    test_dataset = datasets.ImageFolder(root=path_test, transform=transform)

    return train_dataset, test_dataset

def get_data_loaders(train_dataset, test_dataset, batch_size=32):
    """Create data loaders."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Training Functions
def train_epoch(model, train_loader, loss_function, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % 100 == 0 and batch_idx > 0:
            avg_loss = running_loss / 100
            accuracy = 100 * correct / total
            print(f"Batch {batch_idx}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            running_loss = 0.0
            correct = 0
            total = 0

def evaluate(model, test_loader, device):
    """Evaluate the model on test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return 100. * correct / total

def train_model(data_path, model_path, num_epochs=3, batch_size=32, lr=0.001):
    """Complete training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = get_transforms()
    train_dataset, test_dataset = load_datasets(data_path, transform)
    train_loader, test_loader = get_data_loaders(train_dataset, test_dataset, batch_size)

    print(f"Classes: {train_dataset.classes}")
    print(f"Class to idx: {train_dataset.class_to_idx}")

    model = NNClassifier().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_epoch(model, train_loader, loss_function, optimizer, device)
        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {accuracy:.2f}%")

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Inference Functions
def load_model(model_path, device):
    """Load the trained model."""
    model = NNClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_single_image(model, img_path, transform, device, class_names):
    """Predict class for a single image."""
    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    return class_names[predicted_idx.item()], confidence.item()

def run_inference(model_path, image_path, data_path):
    """Run inference on a single image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    model = load_model(model_path, device)
    transform = get_transforms()

    if os.path.exists(image_path):
        predicted_class, confidence = predict_single_image(model, image_path, transform, device, class_names)
        print(f"Prediction: {predicted_class} ({confidence*100:.2f}%)")
    else:
        print(f"Error: Image not found at {image_path}")

# Main Function
def main():
    parser = argparse.ArgumentParser(description="PyTorch Image Classification")
    parser.add_argument("--mode", choices=["train", "infer"], required=True,
                        help="Mode: train or infer")
    parser.add_argument("--data_path", help="Path to dataset (will download if not provided)")
    parser.add_argument("--model_path", default="my_classifier.pth", help="Path to save/load model")
    parser.add_argument("--image_path", help="Path to image for inference")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    # Download dataset if not provided
    if not args.data_path:
        print("Downloading dataset...")
        args.data_path = kagglehub.dataset_download("puneet6060/intel-image-classification")
        print(f"Dataset downloaded to: {args.data_path}")

    if args.mode == "train":
        train_model(args.data_path, args.model_path, args.epochs, args.batch_size, args.lr)
    elif args.mode == "infer":
        if not args.image_path:
            # Default test image
            args.image_path = os.path.join(args.data_path, "seg_pred", "seg_pred", "1018.jpg")
        run_inference(args.model_path, args.image_path, args.data_path)

if __name__ == "__main__":
    main()
