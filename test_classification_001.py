# -*- coding: utf-8 -*-
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow import keras
from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation
from tensorflow.keras.layers import Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split 


X_data = np.load('./X_data.npy')
Y_data = np.load('./Y_data.npy')

x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=10)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.2, random_state=10)
x_train /= 255
x_val /= 255
x_test /= 255



model = Sequential()
#Layer 1
model.add(Conv2D(8, (4,4), strides=(1,1), padding='same',activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D((8,8), strides=(8,8), padding='same'))
#Layer 2
model.add(Conv2D(16, (2,2), strides=(1,1), padding='same',activation='relu'))
model.add(MaxPooling2D((4,4), strides=(4,4), padding='same'))
#Output layer
model.add(Flatten())
model.add(Dense(3,activation='softmax')) #Class 개수
#Compile & run
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=140, batch_size=64, validation_data=(x_val, y_val))

score = model.evaluate(x_test, y_test, batch_size=64)
print('test_loss: ', score[0])
print('test_accuracy: ', score[1])
