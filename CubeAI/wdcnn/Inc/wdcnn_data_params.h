/**
  ******************************************************************************
  * @file    wdcnn_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    Wed Feb  8 14:39:35 2023
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef WDCNN_DATA_PARAMS_H
#define WDCNN_DATA_PARAMS_H
#pragma once

#include "ai_platform.h"

/*
#define AI_WDCNN_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_wdcnn_data_weights_params[1]))
*/

#define AI_WDCNN_DATA_CONFIG               (NULL)


#define AI_WDCNN_DATA_ACTIVATIONS_SIZES \
  { 9280, }
#define AI_WDCNN_DATA_ACTIVATIONS_SIZE     (9280)
#define AI_WDCNN_DATA_ACTIVATIONS_COUNT    (1)
#define AI_WDCNN_DATA_ACTIVATION_1_SIZE    (9280)



#define AI_WDCNN_DATA_WEIGHTS_SIZES \
  { 54880, }
#define AI_WDCNN_DATA_WEIGHTS_SIZE         (54880)
#define AI_WDCNN_DATA_WEIGHTS_COUNT        (1)
#define AI_WDCNN_DATA_WEIGHT_1_SIZE        (54880)



#define AI_WDCNN_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_wdcnn_activations_table[1])

extern ai_handle g_wdcnn_activations_table[1 + 2];



#define AI_WDCNN_DATA_WEIGHTS_TABLE_GET() \
  (&g_wdcnn_weights_table[1])

extern ai_handle g_wdcnn_weights_table[1 + 2];


#endif    /* WDCNN_DATA_PARAMS_H */
