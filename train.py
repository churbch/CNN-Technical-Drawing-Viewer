import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.datasets import ImageFolder
import torch.optim as optim
import json
from model import PartsDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_classes = 200  # Dependant on part numbers, will affect model accuracy as it gets larger
epochs = 400 # Number used on other vision tools, begin with a smaller number and infer checkpoint, then review
batch_size = 8
learning_rate = 1e-4

# Data loading (review)
transforms = Compose([Resize((800, 800)), ToTensor()])
train_dataset = ImageFolder(root='Training Data/Train', transform=transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Loads JSON that holds all image / data label pairings for training
with open('mapping.json') as f:
    class_mapping = json.load(f)

# Model initialization
model = PartsDetector(num_classes=num_classes).to(device)
criterion_detection = nn.MSELoss()  # placeholder, revisit
criterion_completeness = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

""" Loss function needs more research, currently using MSE for detection and BCELoss for completeness.
This code would save a checkpoint per epoch.

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions, completeness = model(images)

        # Dummy placeholders for loss computation
        loss_detection = sum([criterion_detection(p, torch.zeros_like(p)) for p in predictions])
        loss_completeness = criterion_completeness(completeness, torch.zeros_like(completeness))

        loss = loss_detection + loss_completeness
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}')
    torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pth')
    """

# Similar to CSRNet and other similar tools, save a best model and the final model.

best_loss = float('inf')  # Initialize to a very high value

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions, completeness = model(images)

        # Placeholder loss (replace with your actual loss computation)
        loss_detection = sum([criterion_detection(p, torch.zeros_like(p)) for p in predictions])
        loss_completeness = criterion_completeness(completeness, torch.zeros_like(completeness))

        loss = loss_detection + loss_completeness
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

    # Save only if the current model is better
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        torch.save(model.state_dict(), 'checkpoints/best_model.pth')
        print(f'Best model saved at Epoch {epoch+1}')

# Optionally save final checkpoint
torch.save(model.state_dict(), 'checkpoints/final_checkpoint.pth')
