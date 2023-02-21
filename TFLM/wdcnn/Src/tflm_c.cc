/**
 ******************************************************************************
 * @file    tflm_c.cc
 * @author  MCD/AIS Team
 * @brief   Light C-wrapper for C++ TFL for MicroControllers runtime interface
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

/*
 * see header file
 *
 * Notes: (implementation consideration)
 * - when an instance of the interpreter is created, a context object (~300 Bytes)
 *   is created, dynamically allocated (see tflm_c_create() fct).
 *   This allocation is done, through the new C++ operator. To be able to
 *   monitor the usage of the heap, new operator is expected to be
 *   based on the C-malloc/free functions.
 */

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_profiler.h"
// #include "tensorflow/lite/micro/simple_memory_allocator.h"
#include "tensorflow/lite/micro//memory_helpers.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"

// #include "tensorflow/lite/version.h"
// Version is forced here, because current tflite-micro github is not tagged
//  2.7.0 here, because, src is aligned on the TF 2.7.0 github.
#define TF_MAJOR_VERSION      2
#define TF_MINOR_VERSION      10
#define TF_PATCH_VERSION      0

#include "tflm_c.h"

// if (=0), resolver is created with all built-in operators
#if !defined(TFLM_RUNTIME_USE_ALL_OPERATORS)
#define TFLM_RUNTIME_USE_ALL_OPERATORS 1
#endif

// forward declaration
class CTfLiteInterpreterContext;

class CTfLiteProfiler : public tflite::MicroProfiler {
public:
  CTfLiteProfiler(CTfLiteInterpreterContext* interp) : ctx_(interp), options_(nullptr),
    event_starts_(0), event_ends_(0) {}
  ~CTfLiteProfiler() override = default;

  uint32_t BeginEvent(const char* tag) override {
    if (options_)
      return begin_event(tag);
    else
      return 0;
  }

  void EndEvent(uint32_t event_handle) override {
    if (options_)
      end_event(event_handle);
  }

  TfLiteStatus register_cb(struct tflm_c_observer_options* options) {
    /* a get_time cb is mandatory to use the profiler */
    if (options_ || options == nullptr || options->get_time == nullptr)
      return kTfLiteError;
    options_ = options;
    start();
    return kTfLiteOk;
  }

  TfLiteStatus unregister_cb(struct tflm_c_observer_options* options) {
    if (options_ == options) {
      options_ = nullptr;
      return kTfLiteOk;
    }
    return kTfLiteError;
  }

  TfLiteStatus start(void) {
    cb_dur_ = 0;
    node_dur_ = 0;
    node_idx_ = -1;
    node_ts_begin_ = 0;
    event_starts_ = 0;
    event_ends_ = 0;
    return kTfLiteOk;
  }

  TfLiteStatus reset(void) {
    node_idx_ = -1;
    return kTfLiteOk;
  }

  TfLiteStatus info(struct tflm_c_profile_info* p_info) {
    if (!p_info) {
      return kTfLiteError;
    }
    p_info->cb_dur = cb_dur_;
    p_info->node_dur = node_dur_;
    p_info->n_events = event_starts_;
    return kTfLiteOk;
  }

protected:
  uint32_t begin_event(const char* tag);

  void end_event(uint32_t event_handle);

  int event_starts() { return event_starts_; }
  int event_ends() { return event_ends_; }

private:
  CTfLiteInterpreterContext* ctx_;
  struct tflm_c_observer_options* options_;
  int event_starts_;
  int event_ends_;
  uint64_t node_ts_begin_;
  const char* node_tag_;
  int32_t node_idx_;
  uint64_t cb_dur_;
  uint64_t node_dur_;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

class CTfLiteInterpreterContext {
public:
  CTfLiteInterpreterContext(const tflite::Model* model,
      const tflite::MicroOpResolver& op_resolver,
      uint8_t* tensor_arena, size_t tensor_arena_size,
      tflite::ErrorReporter* error_reporter): model_(model), profiler(this),
          interpreter(model, op_resolver, tensor_arena, tensor_arena_size, error_reporter,
              nullptr, (tflite::MicroProfiler *)&profiler),
          n_invoks(0) {}

public:
  const tflite::Model *model_;
  CTfLiteProfiler profiler;
  tflite::MicroInterpreter interpreter;
  int n_invoks;

public:
  static TfLiteStatus input(const uint32_t hdl, int32_t index, struct tflm_c_tensor_info* t_info) {
    CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
    const TfLiteTensor* tens = ctx->interpreter.input(index);
    return ctx->tflitetensor_to(tens, t_info, -1);
  }

  static TfLiteStatus output(const uint32_t hdl, int32_t index, struct tflm_c_tensor_info* t_info) {
    CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
    const TfLiteTensor* tens = ctx->interpreter.output(index);
    return ctx->tflitetensor_to(tens, t_info, -1);
  }

  static TfLiteStatus invoke(const uint32_t hdl) {
    CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
    ctx->profiler.reset();
    ctx->n_invoks++;
    return ctx->interpreter.Invoke();
  }

  static TfLiteStatus reset_all_variables(const uint32_t hdl) {
    CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
    ctx->profiler.reset();
    return ctx->interpreter.ResetVariableTensors();
  }

  static TfLiteStatus observer_register(const uint32_t hdl, struct tflm_c_observer_options* options)
  {
    CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
    return ctx->profiler.register_cb(options);
  }

  static TfLiteStatus observer_unregister(const uint32_t hdl, struct tflm_c_observer_options* options)
  {
    CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
    return ctx->profiler.unregister_cb(options);
  }

  static TfLiteStatus observer_start(const uint32_t hdl)
  {
    CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
    ctx->n_invoks = 0;
    return ctx->profiler.start();
 }

  static TfLiteStatus observer_info(const uint32_t hdl, struct tflm_c_profile_info* p_info)
  {
    CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
    TfLiteStatus res = ctx->profiler.info(p_info);
    if (res == kTfLiteOk)
      p_info->n_invoks = ctx->n_invoks;
    return res;
  }

  uint32_t get_handle() {
    return (uint32_t)this;
  }

private:
  TfLiteStatus tflitetensor_to(const TfLiteTensor* tfls, struct tflm_c_tensor_info* t_info, int32_t idx=-1);

protected:
 friend class CTfLiteProfiler;
};


TfLiteStatus CTfLiteInterpreterContext::tflitetensor_to(const TfLiteTensor* tft,
    struct tflm_c_tensor_info* ti, int32_t idx)
{
  if (!tft || !ti)
    return kTfLiteError;

  ti->type = tft->type;
  ti->idx = idx;
  ti->bytes = tft->bytes;

  if (tft->dims->size > TFLM_C_MAX_DIM)
    return kTfLiteError;

  memset(&ti->shape.data, 0, sizeof(uint32_t) * TFLM_C_MAX_DIM);
  ti->shape.size = tft->dims->size;
  for (size_t i=0; i<ti->shape.size; i++) {
    ti->shape.data[i] = tft->dims->data[i];
  }

  ti->depth = 0;
  ti->extension = 0;
  ti->height = 1;
  ti->width = 1;

  // mapping is aligned to STM32 Cube.AI expectation
  if (tft->dims->size == 2) { /* batch + 1d array */
    ti->batch = tft->dims->data[0];
    ti->channels = tft->dims->data[1];
  } else if (tft->dims->size == 3) { /* batch + 2d array */
    ti->batch = tft->dims->data[0];
    ti->height = tft->dims->data[1];
    ti->channels = tft->dims->data[2];
  } else if (tft->dims->size == 4) { /* batch + 3d array */
    ti->batch = tft->dims->data[0];
    ti->height = tft->dims->data[1];
    ti->width =  tft->dims->data[2];
    ti->channels = tft->dims->data[3];
  } else if (tft->dims->size == 5) { /* batch + 4d array */
    ti->batch = tft->dims->data[0];
    ti->height = tft->dims->data[1];
    ti->width =  tft->dims->data[2];
    ti->depth =  tft->dims->data[3];
    ti->channels = tft->dims->data[4];
  } else if (tft->dims->size == 6) { /* batch + 5d array */
    ti->batch = tft->dims->data[0];
    ti->height = tft->dims->data[1];
    ti->width =  tft->dims->data[2];
    ti->depth =  tft->dims->data[3];
    ti->extension =  tft->dims->data[4];
    ti->channels = tft->dims->data[5];
  } else {
    return kTfLiteError;
  }

  ti->data = (void *)tft->data.uint8;

  if (tft->quantization.type == kTfLiteAffineQuantization) {
    TfLiteAffineQuantization *quant = (TfLiteAffineQuantization *)tft->quantization.params;
    ti->scale = *(quant->scale->data);
    ti->zero_point = *(quant->zero_point->data);
  } else {
    ti->scale = 0.0f; // nullptr;
    ti->zero_point = 0; // nullptr;
  }

  return kTfLiteOk;
}

static tflite::MicroErrorReporter micro_error_reporter;


#ifdef __cplusplus
extern "C" {
#endif

TfLiteStatus tflm_c_create(const uint8_t *model_data,
    uint8_t *tensor_arena,
    const uint32_t tensor_arena_size,
    uint32_t *hdl)
{
  TfLiteStatus status;

  if (!hdl || !tensor_arena_size || !tensor_arena || !model_data)
    return kTfLiteError;

  *hdl = 0;

  const tflite::Model* model = ::tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("Invalid expected TFLite model version %d instead %d\r\n",
        (int)model->version(), (int)TFLITE_SCHEMA_VERSION);
    return kTfLiteError;
  }

#if defined(TFLM_RUNTIME_USE_ALL_OPERATORS) && TFLM_RUNTIME_USE_ALL_OPERATORS == 1
  static tflite::AllOpsResolver _resolver;
#else
    static tflite::MicroMutableOpResolver<8> _resolver;
    _resolver.AddQuantize();
    _resolver.AddExpandDims();
    _resolver.AddConv2D();
    _resolver.AddReshape();
    _resolver.AddMaxPool2D();
    _resolver.AddFullyConnected();
    _resolver.AddSoftmax();
    _resolver.AddDequantize();
 #endif

  CTfLiteInterpreterContext *ctx = new CTfLiteInterpreterContext(
      model,
      _resolver,
      tensor_arena,
      tensor_arena_size,
      &micro_error_reporter
  );

  // Allocate the resources
  status = ctx->interpreter.AllocateTensors();
  if (status != kTfLiteOk) {
    printf("AllocateTensors() fails\r\n");
    delete ctx;
    return kTfLiteError;
  }

  // tflite::ErrorReporter* error_reporter = &micro_error_reporter;
  // error_reporter->Report("hello %d\n\t", sizeof(CTfLiteInterpreterContext));
  // error_reporter->Report("hello %d\n\t", sizeof(tflite::MicroProfiler));
  // error_reporter->Report("hello %d\n\t", sizeof(tflite::MicroInterpreter));

  *hdl = ctx->get_handle();

  return kTfLiteOk;
}

TfLiteStatus tflm_c_destroy(uint32_t hdl)
{
  CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
  delete ctx;
  return kTfLiteOk;
}

int32_t tflm_c_inputs_size(const uint32_t hdl)
{
  CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
  return ctx->interpreter.inputs_size();
}

int32_t tflm_c_outputs_size(const uint32_t hdl)
{
  CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
  return ctx->interpreter.outputs_size();
}

TfLiteStatus tflm_c_input(const uint32_t hdl, int32_t index, struct tflm_c_tensor_info *t_info)
{
  return CTfLiteInterpreterContext::input(hdl, index, t_info);
}

TfLiteStatus tflm_c_output(const uint32_t hdl, int32_t index, struct tflm_c_tensor_info *t_info)
{
  return CTfLiteInterpreterContext::output(hdl, index, t_info);
}

TfLiteStatus tflm_c_invoke(const uint32_t hdl)
{
  return CTfLiteInterpreterContext::invoke(hdl);
}

TfLiteStatus tflm_c_reset_all_variables(const uint32_t hdl)
{
  return CTfLiteInterpreterContext::reset_all_variables(hdl);
}

int32_t tflm_c_operators_size(const uint32_t hdl)
{
  CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
  return ctx->model_->subgraphs()->Get(0)->operators()->size();
}

int32_t tflm_c_tensors_size(const uint32_t hdl)
{
  CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
  return ctx->model_->subgraphs()->Get(0)->tensors()->size();
}

int32_t tflm_c_operator_codes_size(const uint32_t hdl)
{
  CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
  return ctx->model_->operator_codes()->size();
}

int32_t tflm_c_arena_used_bytes(const uint32_t hdl)
{
  CTfLiteInterpreterContext *ctx = reinterpret_cast<CTfLiteInterpreterContext *>(hdl);
  return ctx->interpreter.arena_used_bytes();
}

const char* tflm_c_TfLiteTypeGetName(TfLiteType type)
{
  return TfLiteTypeGetName(type);
}

void tflm_c_rt_version(struct tflm_c_version *version)
{
  if (version) {
    version->major = TF_MAJOR_VERSION;
    version->minor = TF_MINOR_VERSION;
    version->patch = TF_PATCH_VERSION;
    version->schema = TFLITE_SCHEMA_VERSION;
  }
}

TfLiteStatus tflm_c_observer_register(const uint32_t hdl, struct tflm_c_observer_options* options)
{
  return CTfLiteInterpreterContext::observer_register(hdl, options);
}

TfLiteStatus tflm_c_observer_unregister(const uint32_t hdl, struct tflm_c_observer_options* options)
{
  return CTfLiteInterpreterContext::observer_unregister(hdl, options);
}

TfLiteStatus tflm_c_observer_start(const uint32_t hdl)
{
  return CTfLiteInterpreterContext::observer_start(hdl);
}

TfLiteStatus tflm_c_observer_info(const uint32_t hdl, struct tflm_c_profile_info* p_info)
{
  return CTfLiteInterpreterContext::observer_info(hdl, p_info);
}

uint32_t CTfLiteProfiler::begin_event(const char* tag)
{
  volatile uint64_t ts = options_->get_time(0);
  event_starts_++;
  node_tag_ = tag;
  node_idx_++;
  node_ts_begin_ = options_->get_time(0);  // ts before node execution
  cb_dur_ += (node_ts_begin_ - ts);
  return 0;
}

void CTfLiteProfiler::end_event(uint32_t event_handle)
{
  volatile uint64_t ts = options_->get_time(0);
  event_ends_++;
  if (options_->notify) {
    struct tflm_c_node node;
    node.node_info.name = node_tag_;
    node.node_info.idx = node_idx_;
    node.node_info.dur = ts - node_ts_begin_;
    node.node_info.n_outputs = 0;
    node.output = nullptr;
    node_dur_ += node.node_info.dur;
    options_->notify(options_->cookie, options_->flags, &node);
  }
  cb_dur_ += (options_->get_time(1) - ts);
}

#ifdef __cplusplus
}
#endif
