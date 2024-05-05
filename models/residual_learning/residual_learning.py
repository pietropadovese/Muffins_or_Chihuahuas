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
    Add,
    GlobalAveragePooling2D,
    DepthwiseConv2D,
    AveragePooling2D
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


def identity_block(X : tf.Tensor, filter : int, block_number : int, layer_rep : int, increase_dim : object) -> tf.Tensor:

    """
    increase_dim : boolean variable to increase the number of dimension of the shortcut
    """

    X_short = X

    # Convultional layer encoding
    for i in range(layer_rep - 1):

        names = [rf"Conv_{block_number}_{i+1}", rf"Batch_{block_number}_{i+1}", rf"ReLu_{block_number}_{i+1}"]

        X = convolutional_block(X, filter, names)

    #After this we need to add one more Conv and Batch before adding
    #X_short and proceeed with the activation

    X = Conv2D(filters = filter,
                kernel_size = 3,
                padding = "Same",
                kernel_initializer = "he_normal",
                name = rf"Conv_{block_number}_{layer_rep}")(X)


    X = BatchNormalization(name = rf"Batch_{block_number}_{layer_rep}")(X)

    if increase_dim:

        X_short = Conv2D(filters = filter,
                        kernel_size = 1,
                        name = rf"short_conv_{block_number}")(X_short)

        X_short = BatchNormalization(name = rf"short_batch_{block_number}")(X_short)


    X = Add()([X, X_short])

    X = ReLU(name = rf"ReLu_{block_number}_{layer_rep}")(X)

    X = MaxPooling2D(name = rf"MaxPool_{block_number}")(X)

    return X


def Preprocessing(inputs):
    
    X = keras.layers.Resizing(224, 224, name = "Resizing")(inputs)
    
    X = keras.layers.Rescaling(scale = 1./255, name = "Rescaling")(X)
    
    X = keras.layers.RandomFlip(mode = "horizontal_and_vertical", name = "Random_Flip")(X)
    
    X = keras.layers.RandomRotation(0.2, name = "Random_Rotation")(X)
    
    return X

class residual_learning:
    
    def create_model():
        
        inputs = keras.Input(shape = (224,224,3))
        X = Preprocessing(inputs)
        X = convolutional_block(X, 16, ['Conv_1', 'Batch_1', 'ReLu1'])
        X = identity_block(X, filter = 16, block_number = 2, layer_rep = 2, increase_dim = False)
        X = identity_block(X, filter = 32, block_number = 3, layer_rep = 3, increase_dim = True)
        X = identity_block(X, filter = 64, block_number = 4, layer_rep = 3, increase_dim = True)
        X = keras.layers.Flatten()(X)
        X = keras.layers.Dropout(0.2)(X)
        X = keras.layers.Dense(512, activation = "ReLU")(X)
        X = keras.layers.Dropout(0.2)(X)
        X = keras.layers.Dense(128, activation = "ReLU")(X)
        outputs = keras.layers.Dense(1, activation = "sigmoid")(X)
        
        
        model = keras.Model(inputs = inputs, outputs = outputs)
        
        return model
    
model = residual_learning.create_model()
model.save("/residual_learning.keras", save_format = "keras")