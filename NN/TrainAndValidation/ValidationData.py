'''
Author: BugCrown
Date: 2023-02-06 19:08:08
LastEditors: BugCrown
LastEditTime: 2023-02-06 19:08:44
FilePath: \TrainAndValidation\ValidationData.py
'''

'''
验证数据（包括输入和标签）的生成，用于microTVM中仿真以及部署
'''
import os
import tensorflow as tf
from WDCNN import WDCNN
from CWRUDataSet import CWRUDataset
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import random
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import preprocessing

from tensorflow.keras.models import save_model, load_model
import sys


os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

tf.random.set_seed(100)

def Gen():
    sample_length = 1024
    overlap = 128
    noise=True
    snr = -10
    val_rate = 0.3
    test_rate = 0.7
    train_dataset = CWRUDataset('../', sample_length = sample_length, overlap = overlap, mode = 'train', val_rate = val_rate, test_rate = test_rate, noise = noise, snr = snr)
    val_dataset = CWRUDataset('../', sample_length = sample_length, overlap = overlap, mode = 'validation', val_rate = val_rate, test_rate = test_rate, noise = noise, snr = snr)
    test_dataset = CWRUDataset('../', sample_length = sample_length, overlap = overlap, mode = 'test', val_rate = val_rate, test_rate = test_rate, noise = noise, snr = snr)    
    
    data_path = "./validation_feature.txt"
    label_path = "./validation_label.txt"
    
    j = 1
    with open(data_path, "w+") as f:
        # for i in val_dataset.feature:
        #     # f.write("[")
        #     for j in i:
        #         f.write(str(j) + " ")
        #         j = j + 1
        #     f.write("\n")
        for i in val_dataset.feature[99]:
            # f.write("[")
            f.write(str(i) + ", ")
            # f.write("\n")
    with open(label_path, "w+") as f:
        for i in val_dataset.label:
            f.write(str(i) + "\n")
        
if __name__ == "__main__":
    Gen()  