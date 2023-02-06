/*
 * @Author: BugCrown
 * @Date: 2023-02-06 16:25:24
 * @LastEditors: BugCrown
 * @LastEditTime: 2023-02-06 17:08:50
 * @FilePath: \TVM_MCU\USER\inc\outputs.h
 */

const size_t output_len = 10;
__attribute__((section(".data.tvm"), aligned(16))) float output[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,};

