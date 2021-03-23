# AgriFresh

Our app is called AgriFresh. It uses advanced machine learning and deep learning. It uses the libraries keras, tensorflow,sklearn, numpy and matplotlib.
We have developed the main classification system. Although the app has yet to have the features to add images of your own as well function for taking pictures (for now we made a website for it, but we'll soon be developing our app). It has a accuracy of 99%.

This app is really useful for the agricultural field. One of the issues that both the farmer as well as the consumer faces is grading of fruits. They both end up taking a lot of time to grade the fruits and guess weather it is fresh and how much it should cost. Our app fixes that problem by grading the fruits quickly for them. Hence saving time and increasing manageability.

If you want to run the HTML file then choose: index.html

If you want to run the python script then choose: python_code.py

To view and download the training and the testing datasets, then choose: training_n_testing_data.txt

NOTE: in case you are using appilaction like juypter notebook or idle then at line 18, 20, 24, 26, 52, 53 replace the given string with the local location of the image files. For example: 'C:\Users\samyak\Desktop\Python Programming\fruits-360\Test/' or 'C:\Users\samyak\Desktop\Python Programming\fruits-360\Training/'

in case you are using google collab then first run the following code:-

from google.colab import drive
drive.mount('/content/drive')

then run,
%mkdir fruits

then run, 
%cd /content/drive

then to make sure the files are loaded run,
%ls MyDrive/fruits-360/Training/ or %ls MyDrive/fruits-360/Test/

after that replace line 18, 20, 24, 26, 52, 53 with 'MyDrive/fruits-360/Training/' or 'MyDrive/fruits-360/Test/'
