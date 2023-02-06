'''
Author: BugCrown
Date: 2023-02-06 18:54:38
LastEditors: BugCrown
LastEditTime: 2023-02-06 18:59:19
FilePath: \Train\CWRUDataSet.py
'''

'''
训练集划分
代码参考百度飞桨相关项目
'''

import os
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import random
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import preprocessing

seed = 102

'''
    故障数据标签
    1   正常数据
    2   内圈故障，故障直径7mil
    3   滚动体故障，故障直径7mil
    4   外圈故障，故障直径7mil
    5   内圈故障，故障直径14mil
    6   滚动体故障，故障直径14mil
    7   外圈故障，故障直径14mil
    8   内圈故障，故障直径21mil
    9   滚动体故障，故障直径21mil
    10   外圈故障，故障直径21mil
'''

FAULT_LABEL_DICT = {'97': 0,
                    '105': 1,
                    '118': 2,
                    '130': 3,
                    '169': 4,
                    '185': 5,
                    '197': 6,
                    '209': 7,
                    '222': 8,
                    '234': 9}

# 选取驱动端数据进行建模
AXIS = '_DE_time'

'''
    sample_length：         样本的长度
    overlap：               两个相邻样本重合的长度
    mode：                  数据集，包括train/eval/test三个
    val_rate：              eval的比例(1-val_rate为train比例)
    test_rate：             eval中有多少比例作为test
    noise：                 是否添加噪声
    snr：                   添加多少分贝的噪声
'''
class CWRUDataset():
    def __init__(self, data_dir, sample_length = 1024, overlap = 128, mode = 'train', val_rate = 0.3, test_rate = 0.5, noise = False, snr = None):
        super(CWRUDataset, self).__init__()
        # self.data_dir = data_dir
        self.sample_length = sample_length
        self.overlap = overlap
        self.mode = mode
        self.noise = noise
        self.snr = snr
        self.feature_all, self.label_all = self.transform(data_dir)

        # 训练集和验证集划分
        train_feature, val_feature, train_label, val_label = train_test_split(self.feature_all, self.label_all, test_size = val_rate, random_state = seed)

        # 标准化
        train_feature, val_feature = self.standardization(train_feature, val_feature)
        
        # 验证集和测试集的划分
        val_feature, test_feature, val_label, test_label = train_test_split(val_feature, val_label, test_size = test_rate, random_state = seed)
        val_feature, test_feature = self.standardization(val_feature, test_feature)
        if self.mode == 'train':
            self.feature = train_feature
            self.label = train_label
        elif self.mode == 'validation':
            self.feature = val_feature
            self.label = val_label
        elif self.mode == 'test':
            self.feature = test_feature
            self.label = test_label
        else:
            raise Exception("mode can only be one of [train, validation, test]")

    # 转换函数，获取数据
    def transform(self, data_dir):
        feature, label = [], []
        for fault_type in FAULT_LABEL_DICT:
            lab = FAULT_LABEL_DICT[fault_type]
            fullaxis = 'X' + fault_type + AXIS
            if fault_type == '97':
                fullaxis = 'X0' + fault_type + AXIS

            matlab_data = loadmat(data_dir + fault_type + '.mat')[fullaxis]
            for i in range(0, len(matlab_data) - self.sample_length, self.overlap):
                sub_matlab_data = matlab_data[i : (i + self.sample_length)].reshape(-1,)
                if self.noise == True:
                    sub_matlab_data = self.awgn(sub_matlab_data, self.snr)
                feature.append(sub_matlab_data)
                label.append(lab)
        return np.array(feature, dtype = "float32"), np.array(label, dtype = "int32")

    # 高斯白噪声
    def awgn(self, data, snr, seed = seed):
        np.random.seed(seed)
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(data ** 2) / len(data)
        npower = xpower / snr
        noise = np.random.randn(len(data)) * np.sqrt(npower)
        return np.array(data + noise)

    # 标准化
    def standardization(self, train_data, val_data):
        scalar = preprocessing.StandardScaler().fit(train_data)
        train_data = scalar.transform(train_data)
        # train_data = np.asarray(train_data).astype("int8")
        # train_data = tuple(train_data)
        val_data = scalar.transform(val_data)
        # val_data = np.asarray(val_data).astype("int8")
        # val_data = tuple(val_data)
        return train_data, val_data          

    def __len__(self):
        return len(self.feature)

    # def __getitem__(self, index):
    #     feature = self.feature[index]
    #     n = int(np.sqrt(len(feature)))
    #     feature = np.reshape(feature, (n, n))
    #     feature = feature[np.newaxis, :]
    #     feature = np.concatenate((feature, feature, feature), axis = 0)

    #     label = self.label[index]
    #     feature = feature.astype("float32")
    #     label = np.array([label], dtype = "int64")
    #     return feature, label

# 与上面的getitem作用一样
def MatrixTrans(input):

    output = input
    n = int(np.sqrt(len(input)))
    output = np.reshape(output, (n, n))
    output = np.expand_dims(output, axis = 2)
    output = np.concatenate((output, output, output), axis = 2)

    return output