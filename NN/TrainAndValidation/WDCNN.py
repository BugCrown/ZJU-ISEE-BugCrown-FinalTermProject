'''
Author: BugCrown
Date: 2023-02-06 18:45:04
LastEditors: BugCrown
LastEditTime: 2023-02-06 18:48:25
FilePath: \Train\WDCNN.py
'''

'''
WDCNN 网络模型
'''

from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Add
from tensorflow.keras.layers import Input, MaxPooling1D, GlobalAveragePooling1D, Flatten
from tensorflow.keras.layers import Dense, Dropout, Softmax
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model

def WDCNN():
    input = Input(shape=(1024, 1))
    x = Conv1D(filters=16, kernel_size=64,strides=8,padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
 
    x = Conv1D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
 
    x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
 
    x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
 
    x = Conv1D(filters=64, kernel_size=3, strides=1, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
 
 
    x = Flatten()(x)
    x = Dense(100)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(10)(x)
    outputs = Activation('softmax')(x)
    model = Model(inputs=input, outputs=outputs)
    return model


if __name__ == '__main__':
    model = WDCNN()
    model.summary()