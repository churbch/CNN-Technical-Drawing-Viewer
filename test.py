import torch
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import json
from model import PartsDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10  # same as trained model

# Load model checkpoint
model = PartsDetector(num_classes=num_classes).to(device)
model.load_state_dict(torch.load('checkpoints/model_epoch_25.pth', map_location=device))
model.eval()

# Load image
transform = Compose([Resize((800,800)), ToTensor()])
image = Image.open('example_test_image.png').convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    predictions, completeness = model(input_tensor)

# Simple result extraction (placeholder for post-processing)
print("Completeness Predictions:", completeness.cpu().numpy())

# To comlete: Metrics calculation, visualization, etc.
# This is a placeholder for the actual metric calculation and visualization code.
