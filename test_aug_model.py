from tensorflow.keras import Model, Input
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation, AvgPool2D, concatenate
import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

import numpy as np
import cv2
import os

# Default values
input_shape = (224, 224)
num_label = 6

epoch = 200
batch = 6
k = 16

# 모델 생성 함수 구현
def create_model_dense():
    '''
    손 제스쳐를 라벨링하는 모델 : color image 버전
    DenseNet 구현
    '''
    inputs = Input(shape=input_shape+(3,))

    x = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(x)

    # DenseNet-121 : 6, 12, 24, 16
    # DenseNet-169 : 6, 12, 32, 32
    # DenseNet-201 : 6, 12, 48, 32
    for repetition in [6, 12, 12, 8]:
        d = dense_block(x, repetition)
        x = transition_layer(d)
    
    x = GlobalAveragePooling2D()(d)
    outputs = Dense(num_label, activation="softmax")(x)

    model = Model(inputs, outputs)

    # Compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def bn_rl_conv(x, filters, kernel=1, strides=1):
    #batch norm + relu + conv
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, kernel, strides=strides, padding='same')(x)
    return x

def dense_block(x, repetition):
    for _ in range(repetition):
        y = bn_rl_conv(x, filters=4*k, kernel=1)
        y = bn_rl_conv(y, filters=k, kernel=3)
        x = concatenate([y, x])
    return x

def transition_layer(x):
    x = bn_rl_conv(x, K.int_shape(x)[-1]//2)
    x = AvgPool2D(2, strides=2, padding='same')(x)
    return x

# 모델 생성
model = create_model_dense()

# Augmentation 구현
train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='constant'
)
valid_aug = ImageDataGenerator(
    rescale=1./255
)

train_gen = train_aug.flow_from_directory(
    './preprocess_dataset/each_sobel25',
    target_size=input_shape,
    batch_size=batch,
    class_mode='categorical'
)
valid_gen = valid_aug.flow_from_directory(
    './preprocess_dataset/each_sobel25_valid',
    target_size=input_shape,
    batch_size=batch,
    class_mode='categorical'
)

# callback 함수
cb_checkpoint = ModelCheckpoint(
    filepath='saved_model/aug_model',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

cb_list = [cb_checkpoint]


# 학습 진행
h = model.fit(
    train_gen,
    epochs=epoch,
    steps_per_epoch=25,
    validation_data=valid_gen,
    validation_steps=1,
    callbacks=cb_list
)
print('Model train done!!\n')


# 테스트해보기
cap = cv2.VideoCapture(0)

print('가즈아아아아아아')
pTime = 0
num_frame = 0
while True:
    success, frame = cap.read()

    if not success: break

    k = 224//2
    frame = frame[320-k:320+k, 240-k:240+k]

    dx3 = cv2.Sobel(frame, cv2.CV_64F, 2, 0, ksize=3)
    dy3 = cv2.Sobel(frame, cv2.CV_64F, 0, 2, ksize=3)

    dx5 = cv2.Sobel(frame, cv2.CV_64F, 2, 0, ksize=5)
    dy5 = cv2.Sobel(frame, cv2.CV_64F, 0, 2, ksize=5)

    sobel3 = cv2.magnitude(dx3, dy3)
    sobel3 = np.clip(sobel3, 0, 255).astype(np.uint8)

    sobel5 = cv2.magnitude(dx5, dy5)
    sobel5 = np.clip(sobel5, 0, 255).astype(np.uint8)

    rate = 0.7
    sobel = cv2.addWeighted(sobel3, 1-rate, sobel5, rate, 0.0)

    # 예측 결과 출력
    labels = ['빠', '주먹', '총', '오케이', '브이', '롹큰롤']
    if num_frame % 5 == 0:
        frame_norm = cv2.addWeighted(sobel, 1/255, frame, 0.0, 0.0)
        result = model.predict(np.expand_dims(frame_norm, axis=0))[0].tolist()
        result_label = labels[result.index(max(result))]
        print(result_label, max(result))


    # 카메라 on
    cv2.imshow('now cam', frame)
    cv2.imshow('sobel', sobel)

    num_frame = (num_frame + 1) % 5

    # ESC 누르면 창 닫힘
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break

cap.release()
cv2.destroyAllWindows()