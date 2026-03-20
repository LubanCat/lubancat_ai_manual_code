// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _RKNN_DEMO_YOLOV5FACE_H_
#define _RKNN_DEMO_YOLOV5FACE_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "common.h"

#include "image_utils.h"

#if defined(RV1106_1103) 
    typedef struct {
        char *dma_buf_virt_addr;
        int dma_buf_fd;
        int size;
    }rknn_dma_buf;
#endif

#define OBJ_NAME_MAX_SIZE 128
#define OBJ_NUMB_MAX_SIZE 512
#define OBJ_CLASS_NUM 1
#define NMS_THRESH 0.5
#define BOX_THRESH 0.6
#define PROP_BOX_SIZE (4 + 1 + 10 + OBJ_CLASS_NUM)

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
#if defined(RV1106_1103) 
    rknn_tensor_mem* input_mems[1];
    rknn_tensor_mem* output_mems[3];
    rknn_dma_buf img_dma_buf;
#endif
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_app_context_t;

typedef struct ponit_t {
    int x;
    int y;
} ponit_t;

typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
    ponit_t ponit[5];
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} yolov5face_result_list;


int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, yolov5face_result_list *od_results);

int init_yolov5face_model(const char* model_path, rknn_app_context_t* app_ctx);

int release_yolov5face_model(rknn_app_context_t* app_ctx);

int inference_yolov5face_model(rknn_app_context_t* app_ctx, image_buffer_t* img, yolov5face_result_list* od_results);

#endif //_RKNN_DEMO_YOLOV5FACE_H_

