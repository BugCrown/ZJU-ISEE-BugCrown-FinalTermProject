'''
Author: BugCrown
Date: 2023-02-06 19:01:38
LastEditors: BugCrown
LastEditTime: 2023-02-06 19:11:00
FilePath: \TrainAndValidation\ConvertModel.py
'''

'''
将Keras模型量化并转换成适合TVM的TensorFlow Lite模型
'''

from tensorflow import keras
import tensorflow as tf

# 装载预训练模型
model = keras.models.load_model('WDCNN.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# 设置优化器
converter.optimizations = [tf.lite.Optimize.DEFAULT]


image_shape = [1,1024,1]

def representative_dataset_gen():
  for _ in range(1000):
    input = tf.random.normal(image_shape)
    yield [input]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32  # or tf.uint8
converter.inference_output_type = tf.float32  # or tf.uint8
tflite_quant_model = converter.convert()
open("WDCNNQuant_1.tflite", "wb").write(tflite_quant_model)