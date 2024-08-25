# Brian Lesko
# 06/09/2024
# Classify Images using VGG16 in a web app, fetch the camera feed, preprocess the image, and make a prediction in real-time.

import streamlit as st
import torch
import numpy as np
import torchvision.models as models # contains the the VGG16 pretrained network.
import torchvision.transforms as transforms
import cv2
from PIL import Image
import time
from customize_gui import gui
import torch.nn as nn
gui = gui()
gui.clean_format(wide=True)
gui.about(author="Brian", text="In this code we classify images using a VGG16 model.")

st.title('Image Classification: Thumbs Up vs Down')

# Load the custom model
model = models.vgg16()  # Create a new model

# Modify the classifier
model.classifier = nn.Sequential(
    nn.Linear(25088, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

model.load_state_dict(torch.load('model.pth'))  # Load the weights

# Preprocess the image to fit the model
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])  

model.eval()

# Placeholders for image and predicion
col1, col2, col3, col4 = st.columns([.5, 6, 2, .5])
with col2:
    st.write("Camera Feed")
    ImageSpot = st.empty()
with col3:
    st.write("Preprocessed Image")
    ImageSpot2 = st.empty()
    Prediction = st.empty()
    TimePlaceholder = st.empty()
    Time2Placeholder = st.empty()
    Time3Placeholder = st.empty()

# Use opencv to get the current camera frame
camera = cv2.VideoCapture(0)
#camera.set(cv2.CAP_PROP_FPS, 20) # FPS
# VGA
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUY2')) # faster than MJPG

count=0
start_time = time.time()
while True: 
    count = count+1
    ret, frame = camera.read()
    ret, jpeg = cv2.imencode('.jpg', frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # show the input image
    frameo = cv2.resize(frame, None, fx=2, fy=2)
    ImageSpot.image(frameo, channels="RGB")

    # Preprocess the image
    time1 = time.time()
    image = Image.fromarray(frame)  # convert OpenCV image to PIL image
    image = preprocess(image)
    # show the preprocess image
    image_disp = transforms.ToPILImage()(image.squeeze(0))
    ImageSpot2.image(image_disp)
    image = image.unsqueeze(0)  # simulate a batch
    Time2Placeholder.write(f'Preprocess Time: {time.time() - time1}')

    # Make a prediction
    time1 = time.time()
    output = model(image)
    Time3Placeholder.write(f'Prediction Time: {time.time() - time1}')

    # Interpret the output
    predicted_class = 1 if output > 0.5 else 0

    # Print the class name
    if predicted_class == 0:
        Prediction.write("Predicted class: thumbs up")
    else:
        Prediction.write("Predicted class: thumbs down")

    TimePlaceholder.write(f'FPS: {count/(time.time()-start_time)}')
