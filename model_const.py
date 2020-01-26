import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten,BatchNormalization,Input,Conv2D, MaxPooling2D,Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from sklearn.model_selection import train_test_split


class Network(object):
    def __init__(self,channels,patch_width,patch_height):
        middle_activation = 'relu'
        output_activation = 'sigmoid'

        #Functional API
        inputs = Input(shape = (patch_width, patch_height,channels))
        
        x = Conv2D(64,(3,3),padding='same')(inputs)
        #x = BatchNormalization()(x)
        x = Activation(middle_activation)(x)
        x = Conv2D(64,(3,3),padding='same')(x)
        #x = Activation(middle_activation)(x)
        #x = MaxPooling2D(2,2)(x)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation(middle_activation)(x)        
        x = Dense(2)(x) 
        #x = BatchNormalization()(x)

        predictions = Activation(output_activation)(x)        

        #print(predictions)

        self.model = Model(inputs=inputs, outputs=predictions)        

    def get_model(self):
        return self.model
