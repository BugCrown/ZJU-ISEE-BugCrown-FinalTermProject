'''
Author: BugCrown
Date: 2023-02-06 19:11:04
LastEditors: BugCrown
LastEditTime: 2023-02-06 19:13:35
FilePath: \TrainAndValidation\Validation.py
'''

'''
对生成的模型进行验证
代码部分参考相关博客
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
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

tf.random.set_seed(100)

def preprocess(image_name):	# 输入图像预处理
    
    img_data = []
    with open(image_name, "r") as f:
        all = f.readlines()
        for i in range(936):
            img_data.append((all[i].split()))
            
    img_data = np.asarray(img_data).astype("float32")
    img_data = np.expand_dims(img_data, axis=0)
    print(img_data.shape)
    # img_data = np.expand_dims(img_data, axis=2)
    return img_data

WDCNN = tf.lite.Interpreter("WDCNNQuant.tflite")	# 加载tflite模型

WDCNN.allocate_tensors()  # 预先计划张量分配以优化推理
tflife_input_details = WDCNN.get_input_details()  # 获取输入节点的细节
tflife_output_details = WDCNN.get_output_details()  # 获取输出节点的细节
with open("validation_label.txt", "r") as f:
    label = np.float32(f.readlines())

# 加载并处理成输入张量，和keras推理或者tensorflow推理的输入张量一样
input = preprocess("./validation_feature.txt")
correct = 0
total = 936

start_time = time.time()

for times in range(936):

    input_tensor = input[0, times]
    input_tensor = np.expand_dims(input_tensor, axis=0)
    input_tensor = np.expand_dims(input_tensor, axis=2)
    # 填入输入tensor
    WDCNN.set_tensor(tflife_input_details[0]['index'], input_tensor)
    # 运行推理
    WDCNN.invoke()
    # 获取推理结果
    custom = WDCNN.get_tensor(tflife_output_details[0]['index'])

    max_value = -1
    max_index = 9
    custom = list(custom)
    for i in range(10):
        if (custom[0][i] > max_value).any():
            max_value = custom
            max_index = i
    # print(max_index, (max_index==label[time]))
    correct = correct + (max_index==label[times])
end_time = time.time()
total_time = -start_time + end_time
print(f"Total time used in prediction: {total_time} s")

accuracy = correct / total
print(f"Prediction accuracy: {accuracy*100}%")