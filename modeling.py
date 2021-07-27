'''
모델 관련 코드

# ResNet 구현 참조 링크
https://junstar92.tistory.com/110
https://dataplay.tistory.com/27

'''
# from tensorflow.keras.models import Sequential
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Lambda, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D, AvgPool2D
from tensorflow.keras.layers import Flatten, Dense, Concatenate, Add
from tensorflow.keras.losses import Loss
import tensorflow.keras.backend as K

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
# from tensorflow.python.ops.gen_array_ops import pad

import numpy as np
import cv2

import config

class GestureClassification():
    '''
    손 이미지를 라벨링하는 모델
    '''
    def __init__(self, image_size=(224, 224), num_label=6):
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
        Resnet 기반

        Default 값
        input shape : (legnth, height, width, 3)
        output shape : (length, 2)
        '''

        inputs = Input(shape=(self.height, self.width, 3))

        x = inputs
        x_conv = Conv2D(8, 3, padding='same', activation='relu')(x)
        x_conv = Conv2D(8, 3, padding='same', activation='relu')(x_conv)
        x = x_conv

        x_conv = Conv2D(8, 3, padding='same', activation='relu')(x)
        x_conv = Conv2D(8, 3, padding='same', activation='relu')(x_conv)
        x = x + x_conv

        x_conv = Conv2D(8, 3, padding='same', activation='relu')(x)
        x_conv = Conv2D(8, 3, padding='same', activation='relu')(x_conv)
        x = x + x_conv

        x = MaxPooling2D(2)(x)

        x = Flatten()(x)
        # x = Dense(10)(x)
        outputs = Dense(self.label, activation='softmax')(x)

        model = Model(inputs, outputs)

        # Compile
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model

        return model

    def create_model2(self):
        '''
        손 제스쳐를 라벨링하는 모델 생성 v2
        아래 링크의 논문을 참조하였음.
        https://sci-hub.se/https://link.springer.com/article/10.1007/s11042-019-7193-4

        Default 값
        input shape : (length, height, width, 3)
        output shape : (length, )
        '''
        # Conv2D Layer 채널 수
        channel = 10

        inputs = Input(shape=(self.height, self.width, 3))

        x = inputs
        
        # Dual Channel 1 : 원래 이미지 처리
        origin = Conv2D(channel, 7, padding='same', activation='relu')(x)
        origin = MaxPooling2D(2)(origin)
        origin = Conv2D(channel, 7, padding='same', activation='relu')(origin)
        origin = MaxPooling2D(2)(origin)
        origin = Flatten()(origin)

        # Dual Channel 2 : Canny Edge된 이미지 처리
        canny = cv2.Canny(x, 40, 160)
        # canny = Lambda(lambda imgs: canny_edge(imgs))(x)
        canny = Conv2D(channel, 7, padding='same', activation='relu')(canny)
        canny = MaxPooling2D(2)(canny)
        canny = Conv2D(channel, 7, padding='same', activation='relu')(canny)
        canny = MaxPooling2D(2)(canny)
        canny = Flatten()(canny)

        # # Dual Channel 2-2 : Sobel Edge된 이미지 처리
        # dx = cv2.Sobel(x, cv2.CV_32F, 1, 0)
        # dy = cv2.Sobel(x, cv2.CV_32F, 0, 1)

        # canny = cv2.magnitude(dx, dy)
        # canny = np.clip(canny, 0, 255).astype(np.uint8)

        # Dual Channel Layer 합치기
        concate = Concatenate([origin, canny])
        # concate = Dense(30)(concate)
        outputs = Dense(self.label)(concate)

        model = Model(inputs, outputs)

        # Compile
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model

        return model


    ################################################################ ResNext start
    ######
    def create_model_resnext(self):
        '''
        손 제스쳐를 라벨링하는 모델 : color image 버전
        ResNext-50 (32x4d) 구현
        '''
        inputs = Input(shape=(self.height, self.width, 3))

        conv1 = Conv2D(filters=64,kernel_size=(7,7),strides=(2,2),padding="same")(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)

        conv = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same")(conv1)
        
        counts = [[3,[128,128,256]],[4,[256,256,512]],[6,[512,512,1024]],[3,[1024,1024,2048]]]
        for count in counts:
            num = count[0]
            fil_num = count[1]
            for _ in range(num):
                conv = self.group_conv(conv, fil_num)
        conv = GlobalAveragePooling2D()(conv)
        outputs = Dense(self.label, activation="softmax")(conv)

        model = Model(inputs,outputs)
        
        # Compile
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model

        return model

    def group_conv(self, x, count_filter):
        first, second, third = count_filter
        input_x = x
        x = Conv2D(filters=first, kernel_size=(1,1), strides=(1,1), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(groups=32, filters=second, kernel_size=(3,3), strides=(1,1), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=third ,kernel_size=(1,1), strides=(1,1), padding="same")(x)
        x = BatchNormalization()(x)
        
        conx = Conv2D(filters=third, kernel_size=(1,1), strides=(1,1), padding="same")(input_x)
        conx = BatchNormalization()(conx)

        x = Add()([x,conx])
        x = Activation('relu')(x)

        return x
    ######
    ################################################################ ResNext end


    ################################################################ DenseNet start
    ######
    def create_model_dense(self):
        '''
        손 제스쳐를 라벨링하는 모델 : color image 버전
        DenseNet 구현
        '''
        inputs = Input(shape=(self.height, self.width, 3))

        x = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(x)

        for repetition in [6, 12, 24, 16]:
            d = self.dense_block(x, repetition)
            x = self.transition_layer(d)
        
        x = GlobalAveragePooling2D()(d)
        outputs = Dense(self.label, activation="softmax")(x)

        model = Model(inputs, outputs)

        # Compile
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model

        return model

    #batch norm + relu + conv
    def bn_rl_conv(self, x, filters, kernel=1, strides=1):
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, kernel, strides=strides, padding='same')(x)
        return x


    def dense_block(self, x, repetition):
        for _ in range(repetition):
            y = self.bn_rl_conv(x, filters=4*32, kernel=1)
            y = self.bn_rl_conv(y, filters=32, kernel=3)
        return x

    def transition_layer(self, x):
        x = self.bn_rl_conv(x, K.int_shape(x)[-1] //2 )
        x = AvgPool2D(2, strides=2, padding='same')(x)
        return x

    ######
    ################################################################ DenseNet end


    def create_model3(self):
        '''
        손 제스쳐를 라벨링하는 모델 생성
        Resnet 기반

        Default 값
        input shape : (legnth, 300, 400, 3)
        output shape : (length, 2)
        '''

        inputs = Input(shape=(self.height, self.width, 3))

        x = inputs
        x_conv = Conv2D(16, 7, padding='same', activation='relu')(x)
        x_conv = Conv2D(16, 7, padding='same', activation='relu')(x_conv)
        x = x_conv

        x_conv = Conv2D(16, 7, padding='same', activation='relu')(x)
        x_conv = Conv2D(16, 7, padding='same', activation='relu')(x_conv)
        x = x + x_conv

        x = MaxPooling2D(2)(x)

        x_conv = Conv2D(64, 3, padding='same', activation='relu')(x)
        x_conv = Conv2D(64, 3, padding='same', activation='relu')(x_conv)
        x = x_conv

        x_conv = Conv2D(64, 3, padding='same', activation='relu')(x)
        x_conv = Conv2D(64, 3, padding='same', activation='relu')(x_conv)
        x = x + x_conv

        x = MaxPooling2D(2)(x)

        x_conv = Conv2D(128, 3, padding='same', activation='relu')(x)
        x_conv = Conv2D(128, 3, padding='same', activation='relu')(x_conv)
        x = x_conv

        x_conv = Conv2D(128, 3, padding='same', activation='relu')(x)
        x_conv = Conv2D(128, 3, padding='same', activation='relu')(x_conv)
        x = x + x_conv

        x = MaxPooling2D(2)(x)

        x = Flatten()(x)
        # x = Dense(10)(x)
        x_dense = Dense(16, activation='relu')(x)
        x = x_dense
        
        x_dense = Dense(64, activation='relu')(x)
        x = x_dense

        x_dense = Dense(36, activation='relu')(x)
        x = x_dense
        
        outputs = Dense(self.label, activation='softmax')(x)

        model = Model(inputs, outputs)

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

        # 모델 개선이 되지않으면 학습을 조기종료하는 콜백함수
        cb_earlystop = EarlyStopping(

        )
        
        self.callback = [cb_checkpoint, cb_logger, cb_earlystop]

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


class LevenLoss(Loss):
    '''
    레벤슈타인 거리를 기반으로한 손실 함수(Loss function)
    
    아직 미완성된 코드입니다.
    '''
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        '''
        아직 미완성된 코드입니다.
        '''
        error = abs(y_true, y_pred)
        ##########################################
        #
        # 레벤슈타인 거리 계산하는 코드 넣기!!
        #
        ##########################################

        return error


from tensorflow.keras.layers import Layer
class CannyLayer(Layer):
    def __init__(self, ths_min, ths_max):
        super(CannyLayer, self).__init__()
        self.ths_min = ths_min
        self.ths_max = ths_max
    
    def build(self, input_shape):
        self.kernal = self.add_variable("kernel")
        ##############################
        # 미완성
        #
        # 참고링크 : https://nodoudt.tistory.com/42
        ##############################


def canny_edge(imgs):
    print(type(imgs))
    print(type(imgs.numpy()))
    can = imgs.numpy().copy()
    for n, img in enumerate(imgs):
        can[n] = cv2.Canny(img, 40, 160)

    return can