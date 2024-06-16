
# Real Time Image Classifier: Thumbs up vs down

This repository is used for training and testing a Convolutional Neural Network for identifying thumbs up and thumbs down hand gestures. Contains train.py, simple_test.py, and test.py. 

&nbsp;

<div align="center"><img src="preview.gif" width="800"></div>

&nbsp;

train.py
Training was completed in a total of 26.15 minutes with 7 passes over the training set of 2500 images ( each pass is called an epoch). After training, the model identified 2382/2500 of the test images correctly for an accuracy of 95.28%. Accuracy improved on the test set each epoch in this order: 58%, 77%, 84%, 88%, 92%, 94%, and 95% (rounded) while the loss decreased from 0.7721 to 0.1551 during training. Before training yourself, you'll need to download your own dataset and put the images into two folders in the project directory. I named my folders "like" and "dislike". The model I trained used the like and dislike classes of the [Hagrid dataset](https://github.com/hukenovs/hagrid) like and dislike classes only. My model weights are saved in model.pth

simple_test.py
simple test selects a random image and returns the prediction and actual classification. I ran this program 20 times and tallied 16 correct and 4 incorrect identifications. 

test.py
test.py is a gui based program that uses your laptop's webcam as the input to the trained convolutional neural network. The preprossed model input is shown and the classification takes place. On my hardware, a macbook m3, classification averaged ~11 times per second while running the gui program. The code is compatible with USB camera devices and is written in under 100 lines of pure python code. 

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
streamlit run app.py  
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


