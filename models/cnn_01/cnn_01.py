import keras
import tensorflow as tf
from keras import layers
from keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    AveragePooling2D
)

from typing import List


def Preprocessing(inputs):
    
    X = keras.layers.Resizing(224, 224, name = "Resizing")(inputs)
    
    X = keras.layers.Rescaling(scale = 1./255, name = "Rescaling")(X)
    
    X = keras.layers.RandomFlip(mode = "horizontal_and_vertical", name = "Random_Flip")(X)
    
    X = keras.layers.RandomRotation(0.2, name = "Random_Rotation")(X)
    
    return X


class Sequential:
    
    def create_model():
        
        inputs = keras.Input(shape = (224, 224, 3))
        X = Preprocessing(inputs)
        X = Conv2D(16, kernel_size = (5,5), activation = 'tanh', padding = "same", name = "Conv2D_1")(X)
        X = AveragePooling2D(pool_size = (2,2), name = "AvgPoolg_1")(X)
        X = Conv2D(32, kernel_size = (5,5), activation = 'tanh', padding = "same", name = "Conv2D_2")(X)
        X = AveragePooling2D(pool_size = (2,2),name =  "AvgPool_2")(X)
        X = Conv2D(64, kernel_size = (5,5), activation = 'tanh',padding = "same", name = "Conv2D_3")(X)
        X = AveragePooling2D(pool_size = (2,2),name =  "AvgPool_3")(X)
        X = Flatten()(X)
        outputs = Dense(1, activation = 'sigmoid')(X)
        
        model = keras.Model(inputs = inputs, outputs = outputs)
        
        return model
    
model = Sequential.create_model()
model.save("cnn_01.keras", save_format = "keras")

        
        