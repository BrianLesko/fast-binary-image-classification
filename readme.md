
# Real Time Image Classifier: Thumbs up vs down

This repository houses a binary image classifier designed to distinguish between "thumbs up" and "thumbs down" hand gestures. Leveraging the pretrained VGG16 convolutional neural network, the model boasts rapid training and retraining capabilities, suited for deployment on personal computers. The model achieves 95% accuracy on a test set of 2,500 images, with training completed in just over 26 minutes. The project includes three main Python scripts: train.py, simple_test.py, and test.py.

&nbsp;

<div align="center"><img src="preview.gif" width="800" alt="Demo of real-time thumbs up vs. thumbs down image classification using a convolutional neural network on a web UI"></div>

&nbsp;

## Training Summary

The model was trained using a dataset of 2,500 images divided into two categories: "like" and "dislike". The training was executed over seven epochs and took 26.15 minutes, achieving a final test accuracy of 95.28%. The accuracy increased logarithmically, while the loss decreased from 0.7721 to 0.1551 across epochs. This dataset was sourced from the "like" and "dislike" classes of the Hagrid hand gesture dataset. Model weights are stored in model.pth.

If you plan to train your own model, ensure to download and organize your dataset similarly.

The model weights for the thumbs up and down implementation are saved in model.pth.

## Dataset useage

The classifier was trained to recognize hand gestures performed between 0.5 and 4 meters from the camera. The dataset, consisting of 2,500 images, was sampled from the "like" and "dislike" classes of the [Hagrid Dataset](https://github.com/hukenovs/hagrid).

<div align="center"><img src="dataset.jpeg" width="800"></div>

## Simple Validation

The simple_test.py script conducts random image classifications to validate the trained model, which consistently achieved an accuracy rate above 80% across 20 separate runs.

## Deployment

The repository includes a web app that integrates with either a laptop webcam or USB camera device. On an M3 MacBook, the classifier operates at approximately 12 classifications per second, with the entire deployment script written in fewer than 100 lines of Python code.

To run the deployment code, follow the usage instructions below.

&nbsp;

## Usage

To deploy the model locally, execute the following commands:

```
python3 -m venv my_env  
source my_env/bin/activate # or on windows: source my_env\Scripts\activate  
pip install -r requirements.txt  
streamlit run test.py 8000
```

to stop the app, go back to the terminal and press control C

This will start the local Streamlit server, and you can access the chatbot by opening a web browser and navigating to `http://localhost:8000`.

&nbsp;

## Dependencies

`streamlit` - for the web interface  
`pytorch` - to download the Machine learning model and run it, torch and torchvision  
`opencv` - to fetch the camera feed from your device  
`pillow` - python image library for converting from opencv to torchvision format  
`numpy` - for numerical arrays  
`scikit-learn` - to split the training and testing images  

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


