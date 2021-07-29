from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU, AvgPool2D
from tensorflow.keras.layers import BatchNormalization, Dense
from tensorflow.keras.layers import Input
import numpy as np
from tensorflow.python.keras.layers.merge import concatenate
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.keras.models import Model

def densenet(input_shape, n_classes, filters = 32):
    def bn_rl_conv(x, filters, kernel = 1, strides = 1):

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters,kernel, strides=strides, padding = 'same')(x)
        return x

    def dense_block(x, repetition):
        for _ in range(repetition):
            y = bn_rl_conv(x, 4 * filters)
            y = bn_rl_conv(y, filters, 3)
            x = concatenate([y,x])
        return x

    def transition_layer(x):
        x = bn_rl_conv(x, K.int_shape(x)[-1] //2)
        x = AvgPool2D(2, strides=2, padding = 'same')(x)
        return x
    
    input = Input(input_shape)
    x = Conv2D(64, 7, strides=2, padding='same')(input)
    x = MaxPooling2D(3, strides=2, padding='same')(x)

    for repetition in [6,12,24,16]:
        d = dense_block(x, repetition)
        x = transition_layer(d)

    x = GlobalAveragePooling2D()(d)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input, output)
    return model

input_shape = 224, 224, 3
n_classes = 3

model = densenet(input_shape,n_classes)
model.summary()