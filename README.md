# ZJU-ISEE-BugCrown-FinalTermProject  
此项目为我的2022-2023学年毕业设计，项目名称为端侧神经网络的部署与优化（Deployment and Optimization of Endpoint Neural Networks），此项目仅为demo。项目采用的微控制器均为STM32F407ZGT6。模型于CubeAI文件夹中可见。
# 项目构成
+ NN：神经网络的训练与数据集的处理，本项目数据集使用凯斯西储大学CWRU数据集，神经网络使用WDCNN，使用TensorFlow框架搭建和训练。
+ microTVM：基于TVM-v0.10.0搭建的用于基于Cortex-M4内核的微控制器的应用实例。
+ Keil：基于MDK5和microTVM搭建的项目实例。
+ CubeAI：基于STM32CubeMX、X-Cube-AI和MDK5搭建的项目实例。
+ TFLM：基于STM32CubeMX、X-Cube-AI、TensoFlow Lite Micro和MDK5搭建的项目实例。
# 结果 O2
|  部署框架  |       使用模型       |   PC推理精度   | 板上推理精度 | 板上推理延迟 |
| :-------: |:-------------------: | :-----------: | :---------: | :---------: |
| MicroTVM  | WDCNNQuant_v1.tflite |     97.4%     |    97.4%    |   16.0ms    |
| X-Cube-AI | WDCNNQuant_v1.tflite |     97.4%     |    97.4%    |   22.6ms    |
| X-Cube-AI | WDCNN.tflite         |     99.8%     |    99.8%    |   40.5ms    |
| X-Cube-AI | WDCNN.h5             |     99.8%     |    99.8%    |   63.7ms    |
| TFLM      | WDCNNQuant_v1.tflite |     97.4%     |    97.4%    |   22.6ms   |  

**推理延迟均为运行推理函数（run/invoke等）1000次并进行计时，取得平均推理时间**
