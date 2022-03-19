# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:18:49 2022

@author: Y. Güray Hatipoglu
"""

### Part 1: Deeply Neural Networking with EuroStat
# Very greatly benefited from 
# https://github.com/AnushkaMishra29/Eurosat-tensorflow-/blob/master/eurosat.ipynb
# AnushkaMishra29

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


splits=('train[:80%]','train[80%:]')  # Train/test separation

# Downloading 89.91 MB datafile at first instance
(dataset_raw_train,dataset_raw_test),metadata = tfds.load(name='eurosat', 
                                                          as_supervised=True, 
                                                          split=splits, 
                                                          with_info=True,)

len(dataset_raw_train)  # Length of training data
len(dataset_raw_test)  # Length of test data

get_label_name=metadata.features['label'].int2str
image_size=metadata.features['image'].shape[0]
image_shape = metadata.features['image'].shape
num_class=metadata.features['label'].num_classes

def preprocess(image,label):
  image=tf.cast(image,tf.float32)
  image=image/255
  image=tf.image.resize(image,[image_size,image_size])
  return image, label

training_data=dataset_raw_train.map(preprocess)
validation_data=dataset_raw_test.map(preprocess)

def preview_dataset(dataset):
    plt.figure(figsize=(20,20))
    plot_index=0
    for features in dataset.take(20):
        (image, label) = features
        plot_index += 1
        plt.subplot(5, 4, plot_index)
        # plt.axis('Off')
        label = get_label_name(label.numpy())  #appending label to numpy array
        plt.title( label)
        plt.imshow(image.numpy())
preview_dataset(training_data)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Convolution2D(input_shape=image_shape,filters=32,kernel_size=2,activation=tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))
model.add(tf.keras.layers.Convolution2D(filters=32,kernel_size=2,strides=(2,2),activation=tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(tf.keras.layers.Convolution2D(filters=64,kernel_size=3,activation=tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1)))
model.add(tf.keras.layers.Convolution2D(filters=128,kernel_size=3,activation=tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=512,activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(units=num_class,activation=tf.keras.activations.softmax))

BATCH_SIZE = 32
train_batches = training_data.batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_data.batch(BATCH_SIZE).prefetch(1)

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

EPOCHS =3
history = model.fit(train_batches, epochs=EPOCHS,
                    validation_data=validation_batches)

def predict_class(model, images, show = True):
  for img in images:
    img = image.load_img(img, target_size=(64,64))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255.                                      
    pred = model.predict(img)
    index = np.argmax(pred)
    if show:
        plt.imshow(img[0])                           
        plt.axis('off')
        plt.title(pred_value)
        plt.show()
        
model.save(my_dir) # This saves model for future use
                   # Tip: create an empty folder and save it there,
                   # it will be more convenient
                   # This my_dir will be the working directory in the 
                   # "core" part of SingleImage Sentinel-2 retrieval 
                   # and dnn application pipeline.