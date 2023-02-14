# Keil
## TVM_MCU
+ CORE：stm32f407启动文件
+ OBJ：项目输出和链接中间文件
+ PROJECT：MDK项目文件
+ SYSTEM：系统文件，包括delay，sys和usart
+ TVM：microTVM生成的文件
+ USER：main函数和一些输入输出等头文件
## Includes
+ .\SYSTEM\usart
+ .\SYSTEM\sys
+ .\SYSTEM\delay
+ .\TVM\codegen\inc
+ .\TVM\runtime\inc
+ .\USER\inc
## 环境
### Windows
+ MDK：μVersion V5.38.0.0
+ ARM Compiler：V6.19
+ C Standard：C90
+ ARM CMSIS：V5.9.0
+ ARM CMSIS-NN：V2.7.2
### Linux
+ TVM：V0.10.1
## 项目参考
[正点原子stm32f407最小系统板开发资料库函数版本](http://47.111.11.73/docs/boards/stm32/zdyz_stm32f407_mini_sysboard.html)  
[TVM](https://github.com/apache/tvm.git)
