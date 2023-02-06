// tvm target: c -keys=arm_cpu,cpu -mcpu=cortex-m4
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_cast_subtract_cast_multiply(int8_t* p0, float* T_multiply, uint8_t* global_const_workspace_30_var, uint8_t* global_workspace_31_var) {
  for (int32_t ax1_outer = 0; ax1_outer < 3; ++ax1_outer) {
    for (int32_t ax1_inner = 0; ax1_inner < 4; ++ax1_inner) {
      if (((ax1_outer * 2) + (ax1_inner >> 1)) < 5) {
        int32_t cse_var_1 = ((ax1_outer * 4) + ax1_inner);
        T_multiply[cse_var_1] = (((float)(((int32_t)p0[cse_var_1]) + 128)) * 3.906250e-03f);
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_divide_round_add_clip_cast_expand_dims(float* p0, int8_t* T_expand_dims, uint8_t* global_const_workspace_2_var, uint8_t* global_workspace_3_var) {
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1024; ++ax0_ax1_fused_ax2_fused) {
    float __1 = roundf((p0[ax0_ax1_fused_ax2_fused] * 2.724069e+01f)) + -4.000000e+00f;
    float __2 = (__1) < (1.270000e+02f) ? (__1) : (1.270000e+02f);
    T_expand_dims[ax0_ax1_fused_ax2_fused] = ((int8_t)((__2) > (-1.280000e+02f) ? (__2) : (-1.280000e+02f)));
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(float* input_1_buffer_var, float* Identity_buffer_var, uint8_t* global_const_workspace_0_var, uint8_t* global_workspace_1_var) {
  void* constant_28_let = (&(global_const_workspace_0_var[54144]));
  void* constant_11_let = (&(global_const_workspace_0_var[58112]));
  void* constant_12_let = (&(global_const_workspace_0_var[43776]));
  void* constant_27_let = (&(global_const_workspace_0_var[54400]));
  void* constant_26_let = (&(global_const_workspace_0_var[54656]));
  void* constant_21_let = (&(global_const_workspace_0_var[55680]));
  void* constant_17_let = (&(global_const_workspace_0_var[56448]));
  void* constant_16_let = (&(global_const_workspace_0_var[56704]));
  void* constant_22_let = (&(global_const_workspace_0_var[55424]));
  void* constant_9_let = (&(global_const_workspace_0_var[57728]));
  void* constant_18_let = (&(global_const_workspace_0_var[31488]));
  void* constant_6_let = (&(global_const_workspace_0_var[49920]));
  void* constant_32_let = (&(global_const_workspace_0_var[52480]));
  void* constant_15_let = (&(global_const_workspace_0_var[56960]));
  void* constant_24_let = (&(global_const_workspace_0_var[19200]));
  void* constant_13_let = (&(global_const_workspace_0_var[57472]));
  void* constant_23_let = (&(global_const_workspace_0_var[55168]));
  void* constant_0_let = (&(global_const_workspace_0_var[51456]));
  void* constant_1_let = (&(global_const_workspace_0_var[58624]));
  void* constant_14_let = (&(global_const_workspace_0_var[57216]));
  void* constant_5_let = (&(global_const_workspace_0_var[58368]));
  void* constant_29_let = (&(global_const_workspace_0_var[53888]));
  void* constant_3_let = (&(global_const_workspace_0_var[58496]));
  void* constant_20_let = (&(global_const_workspace_0_var[55936]));
  void* constant_19_let = (&(global_const_workspace_0_var[56192]));
  void* constant_31_let = (&(global_const_workspace_0_var[53488]));
  void* constant_2_let = (&(global_const_workspace_0_var[58560]));
  void* constant_4_let = (&(global_const_workspace_0_var[58432]));
  void* constant_25_let = (&(global_const_workspace_0_var[54912]));
  void* constant_7_let = (&(global_const_workspace_0_var[57984]));
  void* constant_8_let = (&(global_const_workspace_0_var[57856]));
  void* constant_10_let = (&(global_const_workspace_0_var[58240]));
  void* constant_30_let = (&(global_const_workspace_0_var[0]));
  void* constant_33_let = (&(global_const_workspace_0_var[58688]));
  void* sid_17_let = (&(global_workspace_1_var[2048]));
  void* sid_16_let = (&(global_workspace_1_var[0]));
  void* sid_1_let = (&(global_workspace_1_var[2048]));
  void* sid_25_let = (&(global_workspace_1_var[2048]));
  void* sid_24_let = (&(global_workspace_1_var[0]));
  void* sid_9_let = (&(global_workspace_1_var[2048]));
  void* sid_8_let = (&(global_workspace_1_var[0]));
  void* sid_32_let = (&(global_workspace_1_var[0]));
  void* sid_33_let = (&(global_workspace_1_var[1024]));
  void* sid_40_let = (&(global_workspace_1_var[1536]));
  void* sid_41_let = (&(global_workspace_1_var[0]));
  void* sid_44_let = (&(global_workspace_1_var[192]));
  void* sid_47_let = (&(global_workspace_1_var[0]));
  void* sid_48_let = (&(global_workspace_1_var[16]));
  if (tvmgen_default_fused_divide_round_add_clip_cast_expand_dims(input_1_buffer_var, sid_1_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_0(sid_1_let, constant_0_let, constant_1_let, constant_3_let, constant_5_let, sid_8_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_1(sid_8_let, sid_9_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_2(sid_9_let, constant_6_let, constant_7_let, constant_9_let, constant_11_let, sid_16_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_3(sid_16_let, sid_17_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_4(sid_17_let, constant_12_let, constant_13_let, constant_15_let, constant_17_let, sid_24_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_5(sid_24_let, sid_25_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_6(sid_25_let, constant_18_let, constant_19_let, constant_21_let, constant_23_let, sid_32_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_7(sid_32_let, sid_33_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_8(sid_33_let, constant_24_let, constant_25_let, constant_27_let, constant_29_let, sid_40_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_9(sid_40_let, sid_41_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_10(sid_41_let, constant_30_let, constant_31_let, sid_44_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_11(sid_44_let, constant_32_let, constant_33_let, sid_47_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_12(sid_47_let, sid_48_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_cast_subtract_cast_multiply(sid_48_let, Identity_buffer_var, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  return 0;
}

