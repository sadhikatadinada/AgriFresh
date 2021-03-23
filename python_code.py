from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.datasets import load_files
import matplotlib.image as mpimg
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from keras.utils import np_utils

import os, os.path
training_categories = []
training_count = []
for i in os.listdir("MyDrive/fruits-360/Training/"):
    training_categories.append(i)
    training_count.append(len(os.listdir("MyDrive/fruits-360/Training/"+ i)))

test_categories = []
test_count = []
for i in os.listdir("MyDrive/fruits-360/Test/"):
    test_categories.append(i)
    test_count.append(len(os.listdir("MyDrive/fruits-360/Test/"+ i)))

    
print("Count of Fruits in Training set:", sum(training_count))
print("Count of Fruits in Test set:", sum(test_count))

figure_size = plt.rcParams["figure.figsize"]
figure_size[0] = 40
figure_size[1] = 20
plt.rcParams["figure.figsize"] = figure_size
index = np.arange(len(training_categories))
plt.bar(index, training_count)
plt.xlabel('Fruits', fontsize=25)
plt.ylabel('Count of Fruits', fontsize=25)
plt.xticks(index, training_categories, fontsize=15, rotation=90)
plt.title('Distrubution of Fruits with counts in Training Set', fontsize=35)
plt.show()

index2 = np.arange(len(test_categories))
plt.bar(index2, test_count)
plt.xlabel('Fruits', fontsize=25)
plt.ylabel('Count of Fruits', fontsize=25)
plt.xticks(index2, test_categories, fontsize=15, rotation=90)
plt.title('Distrubution of Fruits with counts in Test Set', fontsize=35)
plt.show()

training_dir = 'MyDrive/fruits-360/Training/'
test_dir = 'MyDrive/fruits-360/Test/'

def load_dataset(data_path):
    data_loading = load_files(data_path)
    files_add = np.array(data_loading['filenames'])
    targets_fruits = np.array(data_loading['target'])
    target_labels_fruits = np.array(data_loading['target_names'])
    return files_add,targets_fruits,target_labels_fruits   
    
x_train, y_train,target_labels = load_dataset(training_dir)
x_test, y_test,_ = load_dataset(test_dir)

no_of_classes = len(np.unique(y_train))
no_of_classes
y_train = np_utils.to_categorical(y_train,no_of_classes)
y_test = np_utils.to_categorical(y_test,no_of_classes)
y_train[0]

x_test,x_valid = x_test[7000:],x_test[:7000]
y_test,y_vaild = y_test[7000:],y_test[:7000]
print('Vaildation X : ',x_valid.shape)
print('Vaildation y :',y_vaild.shape)
print('Test X : ',x_test.shape)
print('Test y : ',y_test.shape)

def convert_image_to_array_form(files):
    images_array=[]
    for file in files:
        images_array.append(img_to_array(load_img(file)))
    return images_array

x_train = np.array(convert_image_to_array_form(x_train))
print('Training set shape : ',x_train.shape)

x_valid = np.array(convert_image_to_array_form(x_valid))
print('Validation set shape : ',x_valid.shape)

x_test = np.array(convert_image_to_array_form(x_test))
print('Test set shape : ',x_test.shape)

print('1st training image shape ',x_train[0].shape)

x_train = x_train.astype('float32')/255
x_valid = x_valid.astype('float32')/255
x_test = x_test.astype('float32')/255

def main_model():
    model = Sequential()
    model.add(Conv2D(filters = 16, kernel_size = 2,input_shape=(100,100,3),padding='same')) #step2
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=2)) 
    model.add(Conv2D(filters = 32,kernel_size = 2,activation= 'relu',padding='same')) 
    model.add(MaxPooling2D(pool_size=2)) 
    model.add(Conv2D(filters = 64,kernel_size = 2,activation= 'relu',padding='same')) 
    model.add(MaxPooling2D(pool_size=2)) 
    model.add(Conv2D(filters = 128,kernel_size = 2,activation= 'relu',padding='same')) 
    model.add(MaxPooling2D(pool_size=2)) 
    model.add(Dropout(0.3)) 
    model.add(Flatten())
    model.add(Dense(150)) 
    model.add(Activation('relu'))
    model.add(Dropout(0.4)) 
    model.add(Dense(81,activation = 'softmax')) 
    return model 
model = main_model() 
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(x_train,y_train,
        batch_size = 32,
        epochs=30,
        validation_data=(x_valid, y_vaild),
        verbose=2, shuffle=True)
        
print('\n', 'Test accuracy:', acc_score[1])
