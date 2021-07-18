'''
손 이미지를 분류하는 모델

input  : hand image
output : gesture label
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from setting import config


def create_model():
    '''
    손 제스쳐를 라벨링하는 모델 생성
    input shape : (224, 224, 3)
    output shape : (2, )
    '''
    model = Sequential()

    # Layer 1
    model.add(Conv2D(8, (4, 4), strides=(1, 1), padding='same', activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((8, 8), strides=(8, 8), padding='same'))

    # Layer 2
    model.add(Conv2D(16, (2,2), strides=(1,1), padding='same',activation='relu'))
    model.add(MaxPooling2D((4,4), strides=(4,4), padding='same'))

    # Output layer
    model.add(Flatten())
    model.add(Dense(3,activation='softmax')) #Class 개수

    # Compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def create_callback(model_name='gesture_model', log_name='saved_log'):

    callback_list = [
        # callback : save model
        ModelCheckpoint(
            filepath=config.get_model_path(f'{model_name}.ckpt'),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            save_freq='epoch',
        ),

        # callback : save loss data
        CSVLogger(filename=config.get_log_path(f'{log_name}.csv'))
    ]

    return callback_list

