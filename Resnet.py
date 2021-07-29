import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, ReLU



EPOCHS = 10

class ResidualUnit(Model):
    def __init__(self, filter_in, filter_out, kernel_size):
        super(ResidualUnit,self).__init__()
         