
# from tensorflow.keras import Input, Model
# from tensorflow.keras.layers import Conv2D, Lambda, BatchNormalization, Activation
# from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D, AvgPool2D
# from tensorflow.keras.layers import Flatten, Dense, Concatenate, Add

# from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import os
import numpy as np
import cv2



datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    vertical_flip=True,
    fill_mode='nearest'
)

# datagenset = datagen.flow_from_directory(
#     './dataset/wonki',
#     target_size=(224, 224, 3),
#     class_mode='categorical'
# )

# print(datagenset)

img = load_img('dataset/wonki/원기.jpg')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

datagenset = datagen.flow(
    x, batch_size=1,
    save_to_dir='dataset', save_prefix='won', save_format='jpg'
)

import matplotlib.pyplot as plt

plt.figure(figsize=(4, 5))

i = 1
for n, batch in enumerate(datagenset):
    print(batch.shape, n)
    ax = plt.subplot(4, 5, i)
    i += 1
    plt.imshow((batch[0]*255).astype('uint8'))
    if i > 20:
        break





