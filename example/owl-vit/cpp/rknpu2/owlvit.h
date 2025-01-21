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


#ifndef _RKNN_DEMO_YOLO_WORLD_H_
#define _RKNN_DEMO_YOLO_WORLD_H_

#include "rknn_api.h"
#include "common.h"
#include "image_utils.h"

#define CNT_PRED_BOXES 576
#define LEN_IMAGE_FEATURE 24*24*768
#define  LEN_TEXT_TOKEN 16

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define BOX_THRESH 0.1

typedef struct {
    image_rect_t box;
    float prop;
    int text_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

typedef struct {
    rknn_context owlvit_text_ctx;
    rknn_context owlvit_image_ctx;

    rknn_input_output_num owlvit_text_io_num;
    rknn_tensor_attr* owlvit_text_input_attrs;
    rknn_tensor_attr* owlvit_text_output_attrs;

    rknn_input_output_num owlvit_image_io_num;
    rknn_tensor_attr* owlvit_image_input_attrs;
    rknn_tensor_attr* owlvit_image_output_attrs;

    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_owlvit_context_t;

int init_owlvit_model(rknn_owlvit_context_t* app_ctx, const char* text_model_path, const char* image_model_path);

int release_owlvit_model(rknn_owlvit_context_t* app_ctx);

int inference_owlvit_model(rknn_owlvit_context_t* app_ctx, image_buffer_t* img, char** text_input, int text_nums, object_detect_result_list* od_results);

#endif