#ifndef _RKNN_PICODET_DEMO_POSTPROCESS_H_
#define _RKNN_PICODET_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include <array>
#include "rknn_api.h"
#include "image_utils.h"

#define OBJ_CLASS_NUM 80

#define NMS_TOP_K 100
#define NMS_THRESH 0.45
#define BOX_THRESH 0.5

#define OBJ_NUMB_MAX_SIZE 64
#define NORMALIZED false

typedef struct {
  std::vector<std::array<float, 4>> boxes;
  std::vector<float> scores;
  std::vector<int32_t> label_ids;
} object_detect_result;

int init_post_process();
void deinit_post_process();
char *coco_cls_to_name(int cls_id);
int picodet_post_process(rknn_app_context_t *app_ctx, const float* boxes_data, const float* scores_data, object_detect_result* results);
#endif //_RKNN_PICODET_DEMO_POSTPROCESS_H_
