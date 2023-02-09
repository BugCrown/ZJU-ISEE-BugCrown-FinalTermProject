/**
  ******************************************************************************
  * @file    wdcnn.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Wed Feb  8 14:39:35 2023
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "wdcnn.h"
#include "wdcnn_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_wdcnn
 
#undef AI_WDCNN_MODEL_SIGNATURE
#define AI_WDCNN_MODEL_SIGNATURE     "b56a33e2df0c17f14a1808839d7cc065"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Wed Feb  8 14:39:35 2023"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_WDCNN_N_BATCHES
#define AI_WDCNN_N_BATCHES         (1)

static ai_ptr g_wdcnn_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_wdcnn_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_1_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1024, AI_STATIC)
/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_20_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6528, AI_STATIC)
/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conversion_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)
/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)
/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6528, AI_STATIC)
/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  pool_5_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)
/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1056, AI_STATIC)
/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)
/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  dense_31_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 192, AI_STATIC)
/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  pool_11_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)
/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1088, AI_STATIC)
/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  dense_32_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 100, AI_STATIC)
/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)
/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  pool_17_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)
/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_20_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)
/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_20_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)
/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  pool_23_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)
/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 384, AI_STATIC)
/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  pool_29_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 192, AI_STATIC)
/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  dense_31_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 100, AI_STATIC)
/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  dense_32_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 10, AI_STATIC)
/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  dense_32_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10, AI_STATIC)
/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  nl_33_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 10, AI_STATIC)
/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)
/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)
/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)
/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)
/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6144, AI_STATIC)
/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)
/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_20_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)
/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_20_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)
/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)
/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)
/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  dense_31_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 19200, AI_STATIC)
/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  dense_31_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 100, AI_STATIC)
/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  dense_32_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1000, AI_STATIC)
/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  dense_32_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 10, AI_STATIC)
/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2464, AI_STATIC)
/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3584, AI_STATIC)
/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6144, AI_STATIC)
/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03670978918671608f),
    AI_PACK_INTQ_ZP(-4)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_2_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01748974621295929f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_5_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01748974621295929f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_8_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01748974621295929f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_8_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.018494825810194016f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_11_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.018494825810194016f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_14_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.018494825810194016f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_14_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020797835662961006f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_17_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020797835662961006f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_20_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020797835662961006f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_20_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.047212835401296616f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_23_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.047212835401296616f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_26_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06801117956638336f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_29_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06801117956638336f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_31_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13857808709144592f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_32_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.5422229170799255f),
    AI_PACK_INTQ_ZP(44)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_2_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0016023595817387104f, 0.002017235616222024f, 0.0016058144392445683f, 0.0018076957203447819f, 0.001762905390933156f, 0.0017482673283666372f, 0.0016317382687702775f, 0.0017593791708350182f, 0.001597992959432304f, 0.0021078470163047314f, 0.0011447740253061056f, 0.0013669845648109913f, 0.0019326996989548206f, 0.0016424040077254176f, 0.001765236142091453f, 0.0016362222377210855f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_8_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002610637340694666f, 0.0038288466166704893f, 0.002249430399388075f, 0.0022561477962881327f, 0.0020902815740555525f, 0.001979774096980691f, 0.0022652761545032263f, 0.002340318402275443f, 0.0020950534380972385f, 0.0028108032420277596f, 0.0021315026096999645f, 0.002250598045065999f, 0.0030414971988648176f, 0.0020243851467967033f, 0.0022824490442872047f, 0.0025584653485566378f, 0.002322449581697583f, 0.0026844730600714684f, 0.00196207407861948f, 0.002008463256061077f, 0.0021581891924142838f, 0.0021607328671962023f, 0.0023577255196869373f, 0.0022037068847566843f, 0.0022230525501072407f, 0.0020283462945371866f, 0.002288928022608161f, 0.0019247577292844653f, 0.0021606481168419123f, 0.003623334225267172f, 0.0024381219409406185f, 0.0025144978426396847f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_14_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002152038039639592f, 0.0025178014766424894f, 0.0022183815017342567f, 0.002832582453265786f, 0.0028683484997600317f, 0.0020001966040581465f, 0.0021685641258955f, 0.002840101718902588f, 0.0020353698637336493f, 0.0018111963290721178f, 0.0023423153907060623f, 0.0025629624724388123f, 0.0020064201671630144f, 0.0037641706876456738f, 0.0021651475690305233f, 0.002988032065331936f, 0.00230390508659184f, 0.0017729481915012002f, 0.004028091672807932f, 0.0019306562608107924f, 0.001775261596776545f, 0.0018689503194764256f, 0.0020230773370712996f, 0.0027194670401513577f, 0.0023246686905622482f, 0.003059305716305971f, 0.0018093595281243324f, 0.001686476869508624f, 0.0022153679747134447f, 0.00200883811339736f, 0.0021938623394817114f, 0.0017762017669156194f, 0.0034729682374745607f, 0.002084881067276001f, 0.0023569243494421244f, 0.0019042608328163624f, 0.0022033960558474064f, 0.0019991034641861916f, 0.0019032483687624335f, 0.002547066193073988f, 0.0041966806165874004f, 0.004306674003601074f, 0.0024800291284918785f, 0.001645552460104227f, 0.0019129409920424223f, 0.0024072439409792423f, 0.00226787687279284f, 0.002280246000736952f, 0.0018412955105304718f, 0.0018127403454855084f, 0.0012460547732189298f, 0.0015897401608526707f, 0.0018859674455597997f, 0.0023868591524660587f, 0.0023017944768071175f, 0.0020088350865989923f, 0.0021308695431798697f, 0.0018266256665810943f, 0.0018139826133847237f, 0.0017299032770097256f, 0.0019794509280472994f, 0.002356167882680893f, 0.0017207185737788677f, 0.0019615860655903816f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_20_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0012016461696475744f, 0.0020072998013347387f, 0.0017947005107998848f, 0.0030878945253789425f, 0.0022130380384624004f, 0.0023988124448806047f, 0.003366076620295644f, 0.0032495162449777126f, 0.0018050833605229855f, 0.001346783246845007f, 0.0018146418733522296f, 0.0021260131616145372f, 0.00140645622741431f, 0.0018207962857559323f, 0.002423075959086418f, 0.0012526052305474877f, 0.0021685040555894375f, 0.002451657084748149f, 0.0021895174868404865f, 0.0024792631156742573f, 0.002286206465214491f, 0.0020880692172795534f, 0.0022706007584929466f, 0.0014556177193298936f, 0.003873017616569996f, 0.0019633807241916656f, 0.0018861705902963877f, 0.0026855317410081625f, 0.002014337806031108f, 0.0026838434860110283f, 0.0036326779518276453f, 0.0017654787516221404f, 0.001134004327468574f, 0.0025010090321302414f, 0.0022955338936299086f, 0.002889086725190282f, 0.001957463566213846f, 0.0018988847732543945f, 0.001219589845277369f, 0.0019246727460995317f, 0.0030704347882419825f, 0.0024802163243293762f, 0.003066328587010503f, 0.002452423796057701f, 0.0030898365657776594f, 0.0024790726602077484f, 0.0018153947312384844f, 0.0026053739711642265f, 0.002210439182817936f, 0.0017610307550057769f, 0.0020620364230126143f, 0.0019538514316082f, 0.0026902377139776945f, 0.0018767854198813438f, 0.0026129859033972025f, 0.0024167594965547323f, 0.002320755971595645f, 0.0024784381967037916f, 0.002124165650457144f, 0.0024107128847390413f, 0.0028952767606824636f, 0.0022673574276268482f, 0.0020674276165664196f, 0.0027319900691509247f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_26_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0018042991869151592f, 0.001861827913671732f, 0.0024661323986947536f, 0.002367956330999732f, 0.0026063418481498957f, 0.0021710984874516726f, 0.001294350833632052f, 0.002060794038698077f, 0.0019632684998214245f, 0.002035107696428895f, 0.002285431604832411f, 0.0010626286966726184f, 0.001702296081930399f, 0.00243944488465786f, 0.0024240040220320225f, 0.001678683329373598f, 0.0018952152458950877f, 0.0015020380960777402f, 0.0025852047838270664f, 0.0029480087105184793f, 0.001322774332948029f, 0.0015238545602187514f, 0.002538628876209259f, 0.0020666501950472593f, 0.0020596745889633894f, 0.002111640525981784f, 0.0024597274605184793f, 0.0018632820574566722f, 0.0023115938529372215f, 0.0012109823292121291f, 0.0023893967736512423f, 0.0019207330187782645f, 0.0023142921272665262f, 0.0011829374125227332f, 0.0015365935396403074f, 0.003143400652334094f, 0.0022515759337693453f, 0.002178025431931019f, 0.0010781896999105811f, 0.0021719960495829582f, 0.0022201964166015387f, 0.001553717884235084f, 0.002032182179391384f, 0.001173625816591084f, 0.0019728937186300755f, 0.0016092293662950397f, 0.00234270840883255f, 0.002064105588942766f, 0.002446053083986044f, 0.002411366906017065f, 0.004289975389838219f, 0.0030483778100460768f, 0.0023840784560889006f, 0.0011865603737533092f, 0.0011172514641657472f, 0.0019471837440505624f, 0.0019559417851269245f, 0.0021003722213208675f, 0.0031086646486073732f, 0.002000670414417982f, 0.002035959158092737f, 0.001780236721970141f, 0.001849262393079698f, 0.0017796462634578347f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_31_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006128348875790834f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(dense_32_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0047451225109398365f),
    AI_PACK_INTQ_ZP(0)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  input_1_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1024), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &input_1_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_20_scratch0, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 6528, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6528, 6528),
  1, &conv2d_20_scratch0_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conversion_0_output, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1024), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &conversion_0_output_array, &conversion_0_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conversion_0_output0, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1024, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1024),
  1, &conversion_0_output_array, &conversion_0_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_output, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 128, 1), AI_STRIDE_INIT(4, 1, 1, 16, 2048),
  1, &conv2d_2_output_array, &conv2d_2_output_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_output0, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 128), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &conv2d_2_output_array, &conv2d_2_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_scratch0, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 6528, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6528, 6528),
  1, &conv2d_26_scratch0_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  pool_5_output, AI_STATIC,
  7, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 1, 64), AI_STRIDE_INIT(4, 1, 1, 16, 16),
  1, &pool_5_output_array, &pool_5_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  pool_5_output0, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 64, 1), AI_STRIDE_INIT(4, 1, 1, 16, 1024),
  1, &pool_5_output_array, &pool_5_output_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_pad_before_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 66, 1), AI_STRIDE_INIT(4, 1, 1, 16, 1056),
  1, &conv2d_8_pad_before_output_array, &conv2d_8_pad_before_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_output, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 64, 1), AI_STRIDE_INIT(4, 1, 1, 32, 2048),
  1, &conv2d_8_output_array, &conv2d_8_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_output0, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 64), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &conv2d_8_output_array, &conv2d_8_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  dense_31_scratch0, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 1, 1), AI_STRIDE_INIT(4, 2, 2, 384, 384),
  1, &dense_31_scratch0_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  pool_11_output, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 32), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &pool_11_output_array, &pool_11_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  pool_11_output0, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &pool_11_output_array, &pool_11_output_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_pad_before_output, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 34, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1088),
  1, &conv2d_14_pad_before_output_array, &conv2d_14_pad_before_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  dense_32_scratch0, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 2, 2, 200, 200),
  1, &dense_32_scratch0_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &conv2d_14_output_array, &conv2d_14_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_output0, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 32), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conv2d_14_output_array, &conv2d_14_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  pool_17_output, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 16), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &pool_17_output_array, &pool_17_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  pool_17_output0, AI_STATIC,
  20, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 16, 1), AI_STRIDE_INIT(4, 1, 1, 64, 1024),
  1, &pool_17_output_array, &pool_17_output_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_20_pad_before_output, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 18, 1), AI_STRIDE_INIT(4, 1, 1, 64, 1152),
  1, &conv2d_20_pad_before_output_array, &conv2d_20_pad_before_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_20_output, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 16, 1), AI_STRIDE_INIT(4, 1, 1, 64, 1024),
  1, &conv2d_20_output_array, &conv2d_20_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_20_output0, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 16), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conv2d_20_output_array, &conv2d_20_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  pool_23_output, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 8), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &pool_23_output_array, &pool_23_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  pool_23_output0, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 8, 1), AI_STRIDE_INIT(4, 1, 1, 64, 512),
  1, &pool_23_output_array, &pool_23_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_output, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 6, 1), AI_STRIDE_INIT(4, 1, 1, 64, 384),
  1, &conv2d_26_output_array, &conv2d_26_output_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_output0, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 6), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conv2d_26_output_array, &conv2d_26_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  pool_29_output, AI_STATIC,
  28, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 3), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &pool_29_output_array, &pool_29_output_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  pool_29_output0, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 1, 192, 1, 1), AI_STRIDE_INIT(4, 1, 1, 192, 192),
  1, &pool_29_output_array, &pool_29_output_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  dense_31_output, AI_STATIC,
  30, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 1, 1, 100, 100),
  1, &dense_31_output_array, &dense_31_output_array_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  dense_32_output, AI_STATIC,
  31, 0x1,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 1, 1, 10, 10),
  1, &dense_32_output_array, &dense_32_output_array_intq)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  dense_32_0_conversion_output, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &dense_32_0_conversion_output_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  nl_33_output, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &nl_33_output_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_weights, AI_STATIC,
  34, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 16), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conv2d_2_weights_array, &conv2d_2_weights_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_bias, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_2_bias_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_weights, AI_STATIC,
  36, 0x1,
  AI_SHAPE_INIT(4, 16, 3, 1, 32), AI_STRIDE_INIT(4, 1, 16, 48, 48),
  1, &conv2d_8_weights_array, &conv2d_8_weights_array_intq)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_bias, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_8_bias_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_weights, AI_STATIC,
  38, 0x1,
  AI_SHAPE_INIT(4, 32, 3, 1, 64), AI_STRIDE_INIT(4, 1, 32, 96, 96),
  1, &conv2d_14_weights_array, &conv2d_14_weights_array_intq)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_bias, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_14_bias_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_20_weights, AI_STATIC,
  40, 0x1,
  AI_SHAPE_INIT(4, 64, 3, 1, 64), AI_STRIDE_INIT(4, 1, 64, 192, 192),
  1, &conv2d_20_weights_array, &conv2d_20_weights_array_intq)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_20_bias, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_20_bias_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_weights, AI_STATIC,
  42, 0x1,
  AI_SHAPE_INIT(4, 64, 3, 1, 64), AI_STRIDE_INIT(4, 1, 64, 192, 192),
  1, &conv2d_26_weights_array, &conv2d_26_weights_array_intq)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_bias, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_26_bias_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  dense_31_weights, AI_STATIC,
  44, 0x1,
  AI_SHAPE_INIT(4, 192, 100, 1, 1), AI_STRIDE_INIT(4, 1, 192, 19200, 19200),
  1, &dense_31_weights_array, &dense_31_weights_array_intq)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  dense_31_bias, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &dense_31_bias_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  dense_32_weights, AI_STATIC,
  46, 0x1,
  AI_SHAPE_INIT(4, 100, 10, 1, 1), AI_STRIDE_INIT(4, 1, 100, 1000, 1000),
  1, &dense_32_weights_array, &dense_32_weights_array_intq)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  dense_32_bias, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &dense_32_bias_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_scratch0, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 2464, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2464, 2464),
  1, &conv2d_2_scratch0_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_scratch0, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 1, 3584, 1, 1), AI_STRIDE_INIT(4, 1, 1, 3584, 3584),
  1, &conv2d_8_scratch0_array, NULL)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_scratch0, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 6144, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6144, 6144),
  1, &conv2d_14_scratch0_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_33_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_32_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_33_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_33_layer, 33,
  NL_TYPE, 0x0, NULL,
  nl, forward_sm,
  &nl_33_chain,
  NULL, &nl_33_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_32_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_32_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_32_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_32_0_conversion_layer, 32,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &dense_32_0_conversion_chain,
  NULL, &nl_33_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_32_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_31_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_32_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_32_weights, &dense_32_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_32_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  dense_32_layer, 32,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &dense_32_chain,
  NULL, &dense_32_0_conversion_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_31_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_29_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_31_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_31_weights, &dense_31_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_31_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  dense_31_layer, 31,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &dense_31_chain,
  NULL, &dense_32_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_29_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_26_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_29_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_29_layer, 29,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_29_chain,
  NULL, &dense_31_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_26_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_23_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_26_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_26_weights, &conv2d_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_26_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_26_layer, 26,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &conv2d_26_chain,
  NULL, &pool_29_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_23_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_20_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_23_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_23_layer, 23,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_23_chain,
  NULL, &conv2d_26_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_20_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_20_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_20_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_20_weights, &conv2d_20_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_20_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_20_layer, 20,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &conv2d_20_chain,
  NULL, &pool_23_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 conv2d_20_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_20_pad_before_value, AI_ARRAY_FORMAT_S8,
    conv2d_20_pad_before_value_data, conv2d_20_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_20_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_17_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_20_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_20_pad_before_layer, 20,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &conv2d_20_pad_before_chain,
  NULL, &conv2d_20_layer, AI_STATIC, 
  .value = &conv2d_20_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_17_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_14_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_17_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_17_layer, 17,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_17_chain,
  NULL, &conv2d_20_pad_before_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_14_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_14_weights, &conv2d_14_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_14_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_14_layer, 14,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &conv2d_14_chain,
  NULL, &pool_17_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 conv2d_14_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_14_pad_before_value, AI_ARRAY_FORMAT_S8,
    conv2d_14_pad_before_value_data, conv2d_14_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_14_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_11_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_14_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_14_pad_before_layer, 14,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &conv2d_14_pad_before_chain,
  NULL, &conv2d_14_layer, AI_STATIC, 
  .value = &conv2d_14_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_11_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_11_layer, 11,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_11_chain,
  NULL, &conv2d_14_pad_before_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_8_weights, &conv2d_8_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_8_layer, 8,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &conv2d_8_chain,
  NULL, &pool_11_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 conv2d_8_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_8_pad_before_value, AI_ARRAY_FORMAT_S8,
    conv2d_8_pad_before_value_data, conv2d_8_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_8_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_5_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_8_pad_before_layer, 8,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &conv2d_8_pad_before_chain,
  NULL, &conv2d_8_layer, AI_STATIC, 
  .value = &conv2d_8_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 0, 1, 0, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_5_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_5_layer, 5,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_5_chain,
  NULL, &conv2d_8_pad_before_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_2_weights, &conv2d_2_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_2_layer, 2,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_sssa8_ch,
  &conv2d_2_chain,
  NULL, &pool_5_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(8, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 28, 0, 35), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_0_layer, 0,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &conversion_0_chain,
  NULL, &conv2d_2_layer, AI_STATIC, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 54880, 1, 1),
    54880, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 9280, 1, 1),
    9280, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_WDCNN_IN_NUM, &input_1_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_WDCNN_OUT_NUM, &nl_33_output),
  &conversion_0_layer, 0, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 54880, 1, 1),
      54880, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 9280, 1, 1),
      9280, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_WDCNN_IN_NUM, &input_1_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_WDCNN_OUT_NUM, &nl_33_output),
  &conversion_0_layer, 0, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/


/******************************************************************************/
AI_DECLARE_STATIC
ai_bool wdcnn_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_wdcnn_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    input_1_output_array.data = AI_PTR(g_wdcnn_activations_map[0] + 1024);
    input_1_output_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 1024);
    
    conversion_0_output_array.data = AI_PTR(g_wdcnn_activations_map[0] + 1024);
    conversion_0_output_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 1024);
    
    conv2d_2_scratch0_array.data = AI_PTR(g_wdcnn_activations_map[0] + 2048);
    conv2d_2_scratch0_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 2048);
    
    conv2d_2_output_array.data = AI_PTR(g_wdcnn_activations_map[0] + 4512);
    conv2d_2_output_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 4512);
    
    pool_5_output_array.data = AI_PTR(g_wdcnn_activations_map[0] + 1024);
    pool_5_output_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 1024);
    
    conv2d_8_pad_before_output_array.data = AI_PTR(g_wdcnn_activations_map[0] + 2048);
    conv2d_8_pad_before_output_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 2048);
    
    conv2d_8_scratch0_array.data = AI_PTR(g_wdcnn_activations_map[0] + 3104);
    conv2d_8_scratch0_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 3104);
    
    conv2d_8_output_array.data = AI_PTR(g_wdcnn_activations_map[0] + 0);
    conv2d_8_output_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 0);
    
    pool_11_output_array.data = AI_PTR(g_wdcnn_activations_map[0] + 2048);
    pool_11_output_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 2048);
    
    conv2d_14_pad_before_output_array.data = AI_PTR(g_wdcnn_activations_map[0] + 0);
    conv2d_14_pad_before_output_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 0);
    
    conv2d_14_scratch0_array.data = AI_PTR(g_wdcnn_activations_map[0] + 1088);
    conv2d_14_scratch0_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 1088);
    
    conv2d_14_output_array.data = AI_PTR(g_wdcnn_activations_map[0] + 7232);
    conv2d_14_output_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 7232);
    
    pool_17_output_array.data = AI_PTR(g_wdcnn_activations_map[0] + 0);
    pool_17_output_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 0);
    
    conv2d_20_pad_before_output_array.data = AI_PTR(g_wdcnn_activations_map[0] + 1024);
    conv2d_20_pad_before_output_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 1024);
    
    conv2d_20_scratch0_array.data = AI_PTR(g_wdcnn_activations_map[0] + 2176);
    conv2d_20_scratch0_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 2176);
    
    conv2d_20_output_array.data = AI_PTR(g_wdcnn_activations_map[0] + 0);
    conv2d_20_output_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 0);
    
    pool_23_output_array.data = AI_PTR(g_wdcnn_activations_map[0] + 1024);
    pool_23_output_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 1024);
    
    conv2d_26_scratch0_array.data = AI_PTR(g_wdcnn_activations_map[0] + 1536);
    conv2d_26_scratch0_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 1536);
    
    conv2d_26_output_array.data = AI_PTR(g_wdcnn_activations_map[0] + 0);
    conv2d_26_output_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 0);
    
    pool_29_output_array.data = AI_PTR(g_wdcnn_activations_map[0] + 384);
    pool_29_output_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 384);
    
    dense_31_scratch0_array.data = AI_PTR(g_wdcnn_activations_map[0] + 0);
    dense_31_scratch0_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 0);
    
    dense_31_output_array.data = AI_PTR(g_wdcnn_activations_map[0] + 576);
    dense_31_output_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 576);
    
    dense_32_scratch0_array.data = AI_PTR(g_wdcnn_activations_map[0] + 0);
    dense_32_scratch0_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 0);
    
    dense_32_output_array.data = AI_PTR(g_wdcnn_activations_map[0] + 200);
    dense_32_output_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 200);
    
    dense_32_0_conversion_output_array.data = AI_PTR(g_wdcnn_activations_map[0] + 0);
    dense_32_0_conversion_output_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 0);
    
    nl_33_output_array.data = AI_PTR(g_wdcnn_activations_map[0] + 40);
    nl_33_output_array.data_start = AI_PTR(g_wdcnn_activations_map[0] + 40);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool wdcnn_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_wdcnn_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    conv2d_2_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_weights_array.data = AI_PTR(g_wdcnn_weights_map[0] + 0);
    conv2d_2_weights_array.data_start = AI_PTR(g_wdcnn_weights_map[0] + 0);
    
    conv2d_2_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_bias_array.data = AI_PTR(g_wdcnn_weights_map[0] + 1024);
    conv2d_2_bias_array.data_start = AI_PTR(g_wdcnn_weights_map[0] + 1024);
    
    conv2d_8_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_8_weights_array.data = AI_PTR(g_wdcnn_weights_map[0] + 1088);
    conv2d_8_weights_array.data_start = AI_PTR(g_wdcnn_weights_map[0] + 1088);
    
    conv2d_8_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_8_bias_array.data = AI_PTR(g_wdcnn_weights_map[0] + 2624);
    conv2d_8_bias_array.data_start = AI_PTR(g_wdcnn_weights_map[0] + 2624);
    
    conv2d_14_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_14_weights_array.data = AI_PTR(g_wdcnn_weights_map[0] + 2752);
    conv2d_14_weights_array.data_start = AI_PTR(g_wdcnn_weights_map[0] + 2752);
    
    conv2d_14_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_14_bias_array.data = AI_PTR(g_wdcnn_weights_map[0] + 8896);
    conv2d_14_bias_array.data_start = AI_PTR(g_wdcnn_weights_map[0] + 8896);
    
    conv2d_20_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_20_weights_array.data = AI_PTR(g_wdcnn_weights_map[0] + 9152);
    conv2d_20_weights_array.data_start = AI_PTR(g_wdcnn_weights_map[0] + 9152);
    
    conv2d_20_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_20_bias_array.data = AI_PTR(g_wdcnn_weights_map[0] + 21440);
    conv2d_20_bias_array.data_start = AI_PTR(g_wdcnn_weights_map[0] + 21440);
    
    conv2d_26_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_26_weights_array.data = AI_PTR(g_wdcnn_weights_map[0] + 21696);
    conv2d_26_weights_array.data_start = AI_PTR(g_wdcnn_weights_map[0] + 21696);
    
    conv2d_26_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_26_bias_array.data = AI_PTR(g_wdcnn_weights_map[0] + 33984);
    conv2d_26_bias_array.data_start = AI_PTR(g_wdcnn_weights_map[0] + 33984);
    
    dense_31_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_31_weights_array.data = AI_PTR(g_wdcnn_weights_map[0] + 34240);
    dense_31_weights_array.data_start = AI_PTR(g_wdcnn_weights_map[0] + 34240);
    
    dense_31_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_31_bias_array.data = AI_PTR(g_wdcnn_weights_map[0] + 53440);
    dense_31_bias_array.data_start = AI_PTR(g_wdcnn_weights_map[0] + 53440);
    
    dense_32_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_32_weights_array.data = AI_PTR(g_wdcnn_weights_map[0] + 53840);
    dense_32_weights_array.data_start = AI_PTR(g_wdcnn_weights_map[0] + 53840);
    
    dense_32_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_32_bias_array.data = AI_PTR(g_wdcnn_weights_map[0] + 54840);
    dense_32_bias_array.data_start = AI_PTR(g_wdcnn_weights_map[0] + 54840);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/


AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_wdcnn_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_WDCNN_MODEL_NAME,
      .model_signature   = AI_WDCNN_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 726640,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_bool ai_wdcnn_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_WDCNN_MODEL_NAME,
      .model_signature   = AI_WDCNN_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 726640,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}

AI_API_ENTRY
ai_error ai_wdcnn_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_wdcnn_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_error ai_wdcnn_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
    ai_error err;
    ai_network_params params;

    err = ai_wdcnn_create(network, AI_WDCNN_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE)
        return err;
    if (ai_wdcnn_data_params_get(&params) != true) {
        err = ai_wdcnn_get_error(*network);
        return err;
    }
#if defined(AI_WDCNN_DATA_ACTIVATIONS_COUNT)
    if (activations) {
        /* set the addresses of the activations buffers */
        for (int idx=0;idx<params.map_activations.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
    }
#endif
#if defined(AI_WDCNN_DATA_WEIGHTS_COUNT)
    if (weights) {
        /* set the addresses of the weight buffers */
        for (int idx=0;idx<params.map_weights.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
    }
#endif
    if (ai_wdcnn_init(*network, &params) != true) {
        err = ai_wdcnn_get_error(*network);
    }
    return err;
}

AI_API_ENTRY
ai_buffer* ai_wdcnn_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_buffer* ai_wdcnn_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_handle ai_wdcnn_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_wdcnn_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if (!net_ctx) return false;

  ai_bool ok = true;
  ok &= wdcnn_configure_weights(net_ctx, params);
  ok &= wdcnn_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_wdcnn_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_wdcnn_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_WDCNN_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

