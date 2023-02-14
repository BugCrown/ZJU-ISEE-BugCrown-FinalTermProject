# wdcnn
文件夹内包含了STM32CubeMX工程文件和由其生成的MDK5工程。  
MDK工程使用的是STM32CubeMX自动生成的HAL库文件，根据注释修改和添加代码。
## 环境
### Windows
+ STM32CubeMX：V6.7.0
+ X-Cube-AI：V7.3.0
+ MDK：μVersion V5.38.0.0
+ ARM Compiler：V5.05
+ C Standard：C90
+ ARM CMSIS：V5.9.0
+ ARM CMSIS-NN：V2.7.2
## 注意事项
驱动和中间件部分太大没有上传，可通过STM32CubeMX自己生成。
## Model文件夹
存放h5、未量化tflite模型、全整型量化tflite模型文件，三者在STM32F407ZGT6通过X-Cube-AI实现后推理一次所需时间分别为64ms、40ms、23ms左右（运行100次取平均值）。
