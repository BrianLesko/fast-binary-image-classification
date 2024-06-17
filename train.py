import os
import torch
import time
from PIL import Image
#from torch.utils.data import Dataset, DataLoader
#import torch.nn as nn
#import torch.optim as optim
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import numpy as np

print("Loading the VGG model")
# Load and modify the pre-trained VGG16 model
model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze all the pretrained layers


print("Adding classifier")
# Updating the classifier with the correct input size
input_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(input_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.6),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

print("Initializing classifier weights")
# Initialize weights for the new layers
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.classifier.apply(init_weights)

print("freezing the old layers")
# Optimizer setup to update only the new classifier layers
optimizer = optim.Adam(model.parameters(), lr=0.00005)
criterion = nn.BCELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Defining a Transformation for input images")
# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom dataset class to handle images from two folders
class Dataset(Dataset):
    def __init__(self, images, transform=None):
        self.transform = transform
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, label = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

print("Create the test train split")
# Prepare the dataset and dataloaders
all_images = [(os.path.join('./like', file), 0) for file in os.listdir('./like')] + \
             [(os.path.join('./dislike', file), 1) for file in os.listdir('./dislike')]

train_images, test_images = train_test_split(all_images, test_size=2500, train_size=2500, stratify=[label for _, label in all_images])

train_dataset = Dataset(train_images, transform=transform)
test_dataset = Dataset(test_images, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

print("Starting training")
# Training loop
num_epochs = 7
for epoch in range(num_epochs):
    start_time = time.time()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1), labels.type(torch.float))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs.view(-1) > 0.5).type(torch.float)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}: Loss: {running_loss / (i + 1):.4f}, Accuracy: {correct}/{total} or {accuracy:.2f}%, Time: {(time.time() - start_time):.2f} seconds")

# Save and load the model
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
print("Model loaded from model.pth")

# Select a single image to test the model
image_path = './dislike/bd9998f6-1615-4433-93e5-774aa80671d3.jpg'
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0).to(device)

# Make a prediction
model.eval()
with torch.no_grad():
    output = model(image)
    predicted_class = output.item()
    print(f"Predicted class: {predicted_class}")
    print(f"Actual class: {0}")
    print("Done")