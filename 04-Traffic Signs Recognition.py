#Traffic Signs Recognition 
import matplotlib.image as mpimg 
import os 

from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.preprocessing import image_dataset_from_directory 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img 
from keras.utils.np_utils import to_categorical 
from tensorflow.keras.utils import image_dataset_from_directory 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 
from tensorflow.keras.models import Sequential 
from keras import layers 
from tensorflow import keras 
from tensorflow.keras.layers.experimental.preprocessing import Rescaling 
from sklearn.model_selection import train_test_split 

import matplotlib.pyplot as plt 
import tensorflow as tf 
import pandas as pd 
import numpy as np 
from glob import glob 
import cv2 

import warnings 
warnings.filterwarnings('ignore') 

from zipfile import ZipFile data_path 
= '/content/traffic-sign-dataset-classification.zip' with 
	ZipFile(data_path, 'r') as zip
	: zip.extractall()

dataset = '../content/traffic_Data/DATA'
labelfile = pd.read_csv('labels.csv') 
img = cv2.imread("/content/traffic_Data/DATA/10/010_0011.png") 
plt.imshow(img) 
img = cv2.imread("/content/traffic_Data/DATA/23/023_0001.png") 
plt.imshow(img) 
labelfile.head()
labelfile.tail()

train_ds = tf.keras.preprocessing.image_dataset_from_directory(dataset, validation_split=0.2, 
															subset='training', 
															image_size=( 
																224, 224), 
															seed=123, 
															batch_size=32) 
val_ds = tf.keras.preprocessing.image_dataset_from_directory(dataset, validation_split=0.2, 
															subset='validation', 
															image_size=( 
																224, 224), 
															seed=123, 
															batch_size=32) 

class_numbers = train_ds.class_names
class_names = []
for i in class_numbers:
	class_names.append(labelfile['Name'][int(i)])
	
plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
	ax = plt.subplot(5, 5, i + 1) 
    plt.imshow(images[i].numpy().astype("uint8")) 
    plt.title(class_names[labels[i]]) 
    plt.axis("off") 
  
plt.show() 

data_augmentation = tf.keras.Sequential( 
	[ 
		tf.keras.layers.experimental.preprocessing.RandomFlip( 
			"horizontal", input_shape=(224, 224, 3)), 
		tf.keras.layers.experimental.preprocessing.RandomRotation(0.1), 
		tf.keras.layers.experimental.preprocessing.RandomZoom(0.2), 
		tf.keras.layers.experimental.preprocessing.RandomFlip( 
			mode="horizontal_and_vertical") 
	] 
) 
model = Sequential() 
model.add(data_augmentation) 
model.add(Rescaling(1./255)) 
model.add(Conv2D(128, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2))) 
model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2))) 
model.add(Conv2D(128, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2))) 
model.add(Conv2D(256, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2))) 
model.add(Flatten()) 
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(128, activation='relu')) 
model.add(Dense(len(labelfile), activation='softmax')) 

model.summary()

keras.utils.plot_model(
	model,
	show_shapes=True,
	show_dtype=True,
	show_layer_activations=True
)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
			optimizer='adam', 
			metrics=['accuracy']) 
mycallbacks = [EarlyStopping(monitor='val_loss', patience=5)] 
history = model.fit(train_ds, 
				validation_data=val_ds, 
				epochs=50, 
				callbacks=mycallbacks) 

plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.legend(['loss', 'val_loss'], loc='upper right') 
plt.plot(hist.history['accuracy']) 
plt.plot(hist.history['val_accuracy']) 
plt.legend(['accuracy', 'val_accuracy'], loc='upper right') 
