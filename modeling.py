'''
모델 관련 코드
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import config

class GestureClassification():
    '''
    손 이미지를 라벨링하는 모델
    '''
    def __init__(self, image_size=(300, 400), num_label=2):
        self.height = image_size[0]
        self.width = image_size[1]
        self.label = num_label

        self.train_data = None
        self.valid_data = None

        self.create_model()
        self.create_callback()


    def create_model(self):
        '''
        손 제스쳐를 라벨링하는 모델 생성

        Default 값
        input shape : (300, 400, 3)
        output shape : (2, )
        '''
        model = Sequential()

        # Layer 1
        model.add(Conv2D(8, (4, 4), strides=(1, 1), padding='same', activation='relu', input_shape=(self.height, self.width, 3)))
        model.add(MaxPooling2D((8, 8), strides=(8, 8), padding='same'))

        # Layer 2
        model.add(Conv2D(16, (2,2), strides=(1,1), padding='same',activation='relu'))
        model.add(MaxPooling2D((4,4), strides=(4,4), padding='same'))

        # Output layer
        model.add(Flatten())
        model.add(Dense(self.label, activation='softmax')) #Class 개수

        # Compile
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model

        return model


    def create_callback(self, name_model='gesture_model', name_log='saved_log'):
        '''
        model.fit에서 인자로 들어가는 callbacks 함수들의 리스트를 생성한다.
        '''
        # 모델 저장 콜백함수
        cb_checkpoint = ModelCheckpoint(
            filepath=f'saved_model/{name_model}',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            save_freq='epoch',
        )

        # 학습 경과를 csv로 기록하는 콜백함수
        cb_logger = CSVLogger(filename=f'saved_model/{name_log}.csv')
        
        self.callback = [cb_checkpoint, cb_logger]

        return self.callback


    def put_data(self, x, y):
        '''
        손 이미지 데이터를 받아와 전처리 하여 저장한다.
        x : (data_len, height, width, channel)
        y : (data_len,)         <- 0 이상의 정수로 라벨링 된 것
        '''

        x = x / 255
        y = to_categorical(y, self.label)

        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, shuffle=True)

        self.train_data = (x_train, y_train)
        self.valid_data = (x_valid, y_valid)

        print(f'훈련 데이터 : x={x_train.shape}, y={y_train.shape}')
        print(f'검증 데이터 : x={x_valid.shape}, y={y_valid.shape}')


    def start_train(self, epoch=50, batch=64):
        '''
        손 이미지 데이터로 모델을 학습시킨다.
        '''
        self.model.fit(
            self.train_data[0], self.train_data[1],
            validation_data=self.valid_data,
            epochs=epoch, batch_size=batch,
            callbacks=self.callback,
            verbose=1
        )

        print('\n')
        print('Model Train Done')
        print('\n')

        return self.model


    def load_model(self, name_model='gesture_model'):
        '''
        학습된 모델을 불러오는 함수
        '''
        print('Loading model...', end='')

        self.model.load_weights(f'saved_model/{name_model}')

        print('Done!')


    def predict(self, x_test):
        '''
        입력된 이미지의 분류를 예측하여 결과로 출력하는 함수
        '''
        return self.model.predict(x_test)