import keras
import tensorflow as tf
from keras import layers
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    ReLU,
    MaxPooling2D,
)

from typing import List

def convolutional_block(X: tf.Tensor, filter : int, names : List[str]) -> tf.Tensor: 
    
    conv_n, batch_n, relu_n = names

    X = Conv2D(filters = filter,
                kernel_size = 3,
                padding = "Same",
                kernel_initializer = "he_normal",
                name = conv_n)(X)

    X = BatchNormalization(name = batch_n)(X)

    X = ReLU(name = relu_n)(X)

    return X



def Preprocessing(inputs):
    
    X = keras.layers.Resizing(224, 224, name = "Resizing")(inputs)
    
    X = keras.layers.Rescaling(scale = 1./255, name = "Rescaling")(X)
    
    X = keras.layers.RandomFlip(mode = "horizontal_and_vertical", name = "Random_Flip")(X)
    
    X = keras.layers.RandomRotation(0.2, name = "Random_Rotation")(X)
       
    return X

class Sequential_v02:
    
    def create_model():
        
        inputs = keras.Input(shape = (224,224,3))
        X = Preprocessing(inputs)
        X = convolutional_block(X, 32, ['Conv_1', 'Batch_1', 'ReLu1'])
        X = convolutional_block(X, 32, ['Conv_2', 'Batch_2', 'ReLu2'])
        X = keras.layers.MaxPooling2D()(X)
        X = convolutional_block(X, 64, ['Conv_3', 'Batch_3', 'ReLu3'])
        X = convolutional_block(X, 64, ['Conv_4', 'Batch_4', 'ReLu4'])
        X = keras.layers.MaxPooling2D()(X)
        X = convolutional_block(X, 128, ['Conv_5', 'Batch_5', 'ReLu5'])
        X = keras.layers.MaxPooling2D()(X)
        
        X = keras.layers.Flatten()(X)
        X = keras.layers.Dropout(0.2)(X)
        X = keras.layers.Dense(512, activation = "ReLU")(X)
        X = keras.layers.Dropout(0.2)(X)
        X = keras.layers.Dense(128, activation = "ReLU", name = 'Dense_128')(X)
        outputs = keras.layers.Dense(1, activation = "sigmoid")(X)
        
        model = keras.Model(inputs = inputs, outputs = outputs)
        
        return model
    
model = Sequential_v02.create_model()
model.save("cnn_02.keras", save_format = "keras")
        
        