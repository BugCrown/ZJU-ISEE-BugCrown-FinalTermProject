/**
 ******************************************************************************
 * @file    debug_log_imp.cc
 * @author  MCD/AIS Team
 * @brief   Debug log implementation for TFL for MicroControllers runtime
 ******************************************************************************
 * @attention
 *
 * <h2><center>&copy; Copyright (c) 2019,2021 STMicroelectronics.
 * All rights reserved.</center></h2>
 *
 * This software is licensed under terms that can be found in the LICENSE file in
 * the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */

#include "tensorflow/lite/micro/debug_log.h"

#include <string.h>
#include <stdint.h>

extern "C" int tflm_io_write(const void *buff, uint16_t count);

extern "C" void DebugLog(const char* s)
{
  if (!s)
    return;
    
  size_t sl = strlen(s);
  if (sl)
	  tflm_io_write(s, (uint16_t)sl);
}
