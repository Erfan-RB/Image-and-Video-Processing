#Flower Recognition
import numpy as np 
import pandas as pd 
import cv2 
import matplotlib.pyplot as plt 
from PIL import Image 
from tensorflow.keras import layers 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 
from tensorflow.keras.optimizers import Adam 
import tensorflow as tf 
import os

base_dir = '/flowers/'
img_size = 224
batch = 64

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, 
								zoom_range=0.2, horizontal_flip=True, 
								validation_split=0.2) 

test_datagen = ImageDataGenerator(rescale=1. / 255, 
								validation_split=0.2) 

train_datagen = train_datagen.flow_from_directory(base_dir, 
												target_size=( 
													img_size, img_size), 
												subset='training', 
												batch_size=batch) 
test_datagen = test_datagen.flow_from_directory(base_dir, 
												target_size=( 
													img_size, img_size), 
												subset='validation', 
												batch_size=batch) 

model = Sequential() 
model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', 
                 activation='relu', input_shape=(224, 224, 3))) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
  
  
model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 padding='same', activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 
  
  
model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 padding='same', activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 
  
model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 padding='same', activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 
  
model.add(Flatten()) 
model.add(Dense(512)) 
model.add(Activation('relu')) 
model.add(Dense(5, activation="softmax")) 

model.summary()

keras.utils.plot_model(
    model,
    show_shapes = True,
    show_dtype = True,
    show_layer_activations = True
)

model.compile(optimizer=tf.keras.optimizers.Adam(), 
			loss='categorical_crossentropy', metrics=['accuracy']) 

epochs=30
model.fit(train_datagen,epochs=epochs,validation_data=test_datagen)

from tensorflow.keras.models import load_model
model.save('Model.h5')
saveModel=load_model('Model.h5')

train_datagen.class_indices

from keras.preprocessing import image
list_ = ['Daisy','Danelion','Rose','sunflower', 'tulip'] 
test_image = image.load_img('img.jpg',target_size=(224,224))
plt.imshow(test_image) 
test_image = image.img_to_array(test_image) 
test_image = np.expand_dims(test_image,axis=0) 
result = savedModel.predict(test_image) 
print(result) 
i=0
for i in range(len(result[0])): 
  if(result[0][i]==1): 
    print(list_[i]) 
    break

test_image = image.load_img('img2.jpg',target_size=(224,224))
plt.imshow(test_image) 
test_image = image.img_to_array(test_image) 
test_image = np.expand_dims(test_image,axis=0) 
result = savedModel.predict(test_image) 
print(result) 
i=0
for i in range(len(result[0])): 
  if(result[0][i]==1): 
    print(list_[i]) 
    break
