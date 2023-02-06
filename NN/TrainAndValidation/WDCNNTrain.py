'''
Author: BugCrown
Date: 2023-02-06 18:45:15
LastEditors: BugCrown
LastEditTime: 2023-02-06 19:00:01
FilePath: \Train\WDCNNTrain.py
'''

'''
训练并保存WDCNN模型
训练集划分部分参考百度飞桨相关项目
'''

import os
import tensorflow as tf
from WDCNN import WDCNN
from CWRUDataSet import CWRUDataset
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
import matplotlib.pyplot as plt
import time

import numpy as np
import pandas as pd
import random
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import preprocessing

from visdom import Visdom
from tensorflow.keras.models import save_model, load_model
from keras_flops import get_flops
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

tf.random.set_seed(100)

def train():
    sample_length = 1024
    overlap = 128
    noise=True
    snr = -10
    val_rate = 0.3
    test_rate = 0.7

    # Dataset
    train_dataset = CWRUDataset('./', sample_length = sample_length, overlap = overlap, mode = 'train', val_rate = val_rate, test_rate = test_rate, noise = noise, snr = snr)
    val_dataset = CWRUDataset('./', sample_length = sample_length,overlap = overlap, mode = 'validation', val_rate = val_rate, test_rate = test_rate, noise = noise, snr = snr)
    test_dataset = CWRUDataset('./', sample_length = sample_length, overlap = overlap, mode = 'test', val_rate = val_rate, test_rate = test_rate, noise = noise, snr = snr)

    print(train_dataset.__len__())
    print(val_dataset.__len__())
    print(test_dataset.__len__())
    print(train_dataset.feature.shape)
    print(train_dataset.label.shape)
    print(val_dataset.feature.shape)
    print(val_dataset.label.shape)
    print(test_dataset.feature.shape)
    print(test_dataset.label.shape)

    (x_train, y_train), (x_test, y_test)=(train_dataset.feature, train_dataset.label), (test_dataset.feature, test_dataset.label)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.batch(32)

    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_data = test_data.batch(32)

    sample = next(iter(train_data))
    print('sample:', sample[0].shape, sample[1].shape, tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))

    # Train
    model = WDCNN()
    model.build(input_shape = (None, 1024, 1))
    optimizer = optimizers.Adam(learning_rate = 1e-3)
    model.compile(optimizer = optimizer, loss = 'MSE', metrics = ['accuracy'])
    model.summary()
    flops = get_flops(model, batch_size=32)
    print(f"flops: {flops}")
    print("Training Starts")
    start = time.time()
    
    for epoch in range(50):
        for step,(x,y) in enumerate(train_data):
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y,depth = 10)
                loss = tf.losses.categorical_crossentropy(y_onehot,logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
            if step % 100 == 0:
                print(epoch, step, 'loss', float(loss))
        print(epoch, 'loss', float(loss))
    
        total_num = 0
        total_correct = 0
        for x,y in test_data:
            logits = model(x)
            prob = tf.nn.softmax(logits,axis = 1)
            pred = tf.argmax(prob,axis=1)
            pred = tf.cast(pred,dtype=tf.int32)
            correct = tf.cast(tf.equal(pred,y), dtype = tf.int32)
            correct = tf.reduce_sum(correct)
            total_num += x.shape[0]
            total_correct += int(correct)
        acc = total_correct/total_num

    end = time.time()
    
    total_time = end - start
    print("Training Ends")
    print('Training Cost', total_time, 's')
    
    model.save("WDCNN.h5")
    
if __name__ == '__main__':
    with tf.device('/gpu:0'):
        train()
