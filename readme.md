
# Real Time Image Classifier: Thumbs up vs down

This repository is used for training and testing a Convolutional Neural Network for identifying thumbs up and thumbs down hand gestures. The project includes three main Python scripts: train.py, simple_test.py, and test.py.

&nbsp;

<div align="center"><img src="preview.gif" width="800"></div>

&nbsp;

## Training Summary

The Convolutional Neural Network (CNN) was trained using a dataset of 2,500 images, segmented into two categories: "like" and "dislike". Training was completed in 26.15 minutes across seven epochs, achieving a final accuracy of 95.28% on the test set. Here’s a breakdown of the training progress:

Accuracy improved logarithmically with each epoch. Concurrently the loss function output decreased from 0.7721 to 0.1551. The training dataset was sourced from the "like" and "dislike" classes of the Hagrid dataset. 

If you plan to train your own model, ensure to download and organize your dataset similarly.

Themodel weights were saved in model.pth.

## Validation

simple_test.py  
Performs random image classification tests. Validated the trained model by acheiving over an 80% success rate on 20 runs. 

## Deployment

test.py  
A web app for deploying the trained model. Uses the webcam of your laptop or a usb camera device. On my hardware, a macbook m3, classification averaged ~12 times per second. Written in under 100 lines of pure python code. 

&nbsp;

## Dependencies

`Streamlit` - for the web interface  
`pytorch` - to download the Machine learning model and run it, torch and torchvision  
`opencv` - to fetch the camera feed from your device  
`pillow` - python image library for converting from opencv to torchvision format  
`numpy` - for numerical arrays  
`scikit-learn` - to split the training and testing images  

To run, use the following commands in your terminal:
```
python3 -m venv my_env  
source my_env/bin/activate # or on windows: source my_env\Scripts\activate  
pip install -r requirements.txt  
streamlit run test.py  
```

to stop the app, go back to the terminal and press control C

This will start the local Streamlit server, and you can access the chatbot by opening a web browser and navigating to `http://localhost:8501`.

&nbsp;

<hr>

&nbsp;

<div align="center">



╭━━╮╭━━━┳━━┳━━━┳━╮╱╭╮        ╭╮╱╱╭━━━┳━━━┳╮╭━┳━━━╮
┃╭╮┃┃╭━╮┣┫┣┫╭━╮┃┃╰╮┃┃        ┃┃╱╱┃╭━━┫╭━╮┃┃┃╭┫╭━╮┃
┃╰╯╰┫╰━╯┃┃┃┃┃╱┃┃╭╮╰╯┃        ┃┃╱╱┃╰━━┫╰━━┫╰╯╯┃┃╱┃┃
┃╭━╮┃╭╮╭╯┃┃┃╰━╯┃┃╰╮┃┃        ┃┃╱╭┫╭━━┻━━╮┃╭╮┃┃┃╱┃┃
┃╰━╯┃┃┃╰┳┫┣┫╭━╮┃┃╱┃┃┃        ┃╰━╯┃╰━━┫╰━╯┃┃┃╰┫╰━╯┃
╰━━━┻╯╰━┻━━┻╯╱╰┻╯╱╰━╯        ╰━━━┻━━━┻━━━┻╯╰━┻━━━╯
  


&nbsp;


<a href="https://x.com/TheBrianLesko/status/1124018912268554240"><img src="https://raw.githubusercontent.com/BrianLesko/BrianLesko/main/.socials/svg-grey/x.svg" width="30" alt="X Logo"></a> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <a href="https://github.com/BrianLesko"><img src="https://raw.githubusercontent.com/BrianLesko/BrianLesko/main/.socials/svg-grey/github.svg" width="30" alt="GitHub"></a> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <a href="https://www.linkedin.com/in/brianlesko/"><img src="https://raw.githubusercontent.com/BrianLesko/BrianLesko/main/.socials/svg-grey/linkedin.svg" width="30" alt="LinkedIn"></a>

follow all of these for a cookie :)

</div>


&nbsp;


