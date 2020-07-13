#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 02:30:52 2020

@author: admin
"""

import tensorflow as tf
import keras
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD


import time
import matplotlib.pyplot as plt
import json

import coremltools.models

from PIL import Image as pil_image

import numpy as np
import itertools

from sklearn.metrics import confusion_matrix

import coremltools

def showAllImages(path):
    folders = []
    numberOfPictures = 0

    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for folder in d:
            folders.append(os.path.join(r, folder))    
            
    for f in folders:
        print('*****',os.path.basename(f))
        jpgCounter = len(glob.glob1(f,"*.jpeg"))
        numberOfPictures = numberOfPictures + jpgCounter
        
    print(numberOfPictures)
        

def plot_confusion_matrix(cm, classes,
        normalize=False,
        title='Macierz pomyłek',
        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #import itertools
    plt.figure(figsize=(15,15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Klasa rzeczywista')
    plt.xlabel('Klasa przewidywana') 
    #plt.savefig('cm100dpi.png', dpi=100)
    #plt.savefig('cmt200dpi.png', dpi=200)
    #plt.savefig('cm300dpi.png', dpi=300)
    plt.savefig("cm.svg")
    plt.show()

def evaluateModelPerformance():
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,10))
    #fig, ax = plt.subplots(2, 1, figsize=(10,10))
    ax1.set_title('Model Loss')
    ax1.plot(History.history['loss'])
    ax1.plot(History.history['val_loss'])
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epochs')
    ax1.legend(['train', 'test'])
    
    ax2.set_title('Model Accuracy')
    ax2.plot(History.history['sparse_categorical_accuracy'])
    ax2.plot(History.history['val_sparse_categorical_accuracy'])
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.axis('tight')
    ax2.legend(['train', 'test'])
    
    for ax in fig.get_axes():
        ax.label_outer()
    
    fig.tight_layout()    
    #plt.savefig('test100dpi.png', dpi=100)
    #plt.savefig('test200dpi.png', dpi=200)
    #plt.savefig('test300dpi.png', dpi=300)
    plt.savefig("test.svg")
    plt.show()   

train_set_szczecin_dir = '/Users/admin/Private/Programowanie/GitHub/GoogleImagesDownloader/data/Szczecin2/train'
valid_set_szczecin_dir = '/Users/admin/Private/Programowanie/GitHub/GoogleImagesDownloader/data/Szczecin2/validation'
test_set_szczecin_dir = '/Users/admin/Private/Programowanie/GitHub/GoogleImagesDownloader/data/Szczecin2/test'


test_datagen = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
                                      rotation_range=5,
                                      zoom_range=0.15,
                                      fill_mode="nearest")

train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(
    train_set_szczecin_dir, target_size=(224,224), batch_size=90, class_mode='sparse'
)

valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(
    valid_set_szczecin_dir, target_size=(224,224), batch_size=90, class_mode='sparse'
)

test_batches = test_datagen.flow_from_directory(
    test_set_szczecin_dir, target_size=(224,224), batch_size=90, class_mode='sparse', shuffle=False
)



mobile = MobileNet(weights='imagenet')
x = mobile.layers[-6].output
predictions = Dense(45, activation='softmax')(x)
model = Model(inputs=mobile.input, outputs=predictions)

for layer in model.layers[:-23]:
    layer.trainable = False
    
model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', 
              metrics=[keras.metrics.SparseCategoricalAccuracy()])


start = time.time()
print("Start training model")

History = model.fit(train_batches, steps_per_epoch=8, validation_data=valid_batches,
                   validation_steps=2, epochs=1, verbose=2)

print("End training model")
end = time.time()
print(time.strftime("%H:%M:%S", time.gmtime(end - start)))


evaluateModelPerformance()

cm_plot_labels = []

for x in test_batches.class_indices:
    cm_plot_labels.append(x)
    
predictions = model.predict(test_batches, steps=5, verbose=0)  

test_labels = test_batches.classes

cm = confusion_matrix(test_labels, predictions.argmax(1)) 

plot_confusion_matrix(cm, cm_plot_labels)

#create coreml model
coreml_model = coremltools.converters.keras.convert(model, input_names = 'input_9', image_input_names = 'input_9', output_names = 'Identity', class_labels = cm_plot_labels, image_scale=1/255.0)  
  
coreml_model.save('modelSzczecin.mlmodel')

for index, layer in enumerate(mobile.layers):
    print(index, " : ", layer.name)

