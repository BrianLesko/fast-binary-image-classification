from PIL import Image
from torchvision import models, transforms
import torch
import torch.nn as nn
import os
import random

# Load the model
model = models.vgg16(pretrained=False)  # Set pretrained=False because we're loading our own weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Modify the classifier to match the structure of the model in model.pth
model.classifier = nn.Sequential(
    nn.Linear(25088, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

# Load the weights
model.load_state_dict(torch.load('model.pth'))
model = model.to(device)
print("Model loaded from model.pth")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Select a single image to test the model
like_dir = './like'
dislike_dir = './dislike'

like_images = os.listdir(like_dir)
dislike_images = os.listdir(dislike_dir)
all_images = like_images + dislike_images
random_image = random.choice(all_images)
if random_image in like_images:
    image_dir = like_dir
else:
    image_dir = dislike_dir
image_path = os.path.join(image_dir, random_image)
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0).to(device)

# Make a prediction
model.eval()
with torch.no_grad():
    output = model(image)
    predicted_class = 1 if output > 0.5 else 0
    print(f"Predicted class: {predicted_class}")
    actual_class = 0 if image_dir == like_dir else 1
    print(f"Actual class: {actual_class}")
    print("Done")