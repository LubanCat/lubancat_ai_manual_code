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

#include "yolo26_pose.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#ifndef RKNPU1
#include <Float16.h>
#endif

#include <iostream>
#include <cmath>
#include <algorithm>

#include <set>
#include <vector>

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

static float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

static float unsigmoid(float y) {
    return -1.0 * logf((1.0 / y) - 1.0);
}

inline static int32_t __clip(float val, float min, float max) {
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static uint8_t qnt_f32_to_affine_u8(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    uint8_t res = (uint8_t)__clip(dst_val, 0, 255);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}
static float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}

void softmax(float *input, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum_exp = 0.0;
    for (int i = 0; i < size; ++i) {
        sum_exp += expf(input[i] - max_val);
    }

    for (int i = 0; i < size; ++i) {
        input[i] = expf(input[i] - max_val) / sum_exp;
    }
}

static int process_i8(int8_t *input, int grid_h, int grid_w, int stride,
                      std::vector<float> &boxes, std::vector<float> &boxScores, std::vector<int> &classId, float threshold,
                      int32_t zp, float scale, int index) {
    int input_loc_len = 4;
    int validCount = 0;

    int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);
    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < OBJ_CLASS_NUM; a++) {
                if(input[(input_loc_len + a)*grid_w * grid_h + h * grid_w + w ] >= thres_i8) { //[1,input_loc_len+OBJ_CLASS_NUM,grid_h,grid_w]
                    float box_conf_f32 = deqnt_affine_to_f32(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w ],
                                                 zp, scale);
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = deqnt_affine_to_f32(input[i * grid_w * grid_h + h * grid_w + w], zp, scale);
                    }

                    float x1,y1,w_,h_;
                    x1 = (w + 0.5 - loc[0])*stride;
                    y1 = (h + 0.5 - loc[1])*stride;
                    w_ = (loc[0] + loc[2])*stride;
                    h_ = (loc[1] + loc[3])*stride;
                    boxes.push_back(x1);//x
                    boxes.push_back(y1);//y
                    boxes.push_back(w_);//w
                    boxes.push_back(h_);//h
                    boxes.push_back(float(index + (h * grid_w) + w));//keypoints index
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

static int process_fp32(float *input, int grid_h, int grid_w, int stride,
                      std::vector<float> &boxes, std::vector<float> &boxScores, std::vector<int> &classId, float threshold,
                      int32_t zp, float scale, int index) {
    int input_loc_len =  4;
    int validCount = 0;

    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < OBJ_CLASS_NUM; a++) {
                float box_conf_f32 = input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w ];
                if(box_conf_f32 >= threshold) { //[1,tensor_len,grid_h,grid_w]
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = input[i * grid_w * grid_h + h * grid_w + w];
                    }

                    float x1,y1,w_,h_;
                    x1 = (w + 0.5 - loc[0])*stride;
                    y1 = (h + 0.5 - loc[1])*stride;
                    w_ = (loc[0] + loc[2])*stride;
                    h_ = (loc[1] + loc[3])*stride;
                    boxes.push_back(x1);//x
                    boxes.push_back(y1);//y
                    boxes.push_back(w_);//w
                    boxes.push_back(h_);//h
                    boxes.push_back(float(index + (h * grid_w) + w));//keypoints index
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold,
                 object_detect_result_list *od_results) {

    rknn_output *_outputs = (rknn_output *)outputs;
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;
    memset(od_results, 0, sizeof(object_detect_result_list));
    int index = 0;
    for (int i = 0; i < 3; i++) {
        grid_h = app_ctx->output_attrs[i].dims[2];
        grid_w = app_ctx->output_attrs[i].dims[3];
        stride = model_in_h / grid_h;
        if (app_ctx->is_quant) {
            validCount += process_i8((int8_t *)_outputs[i].buf, grid_h, grid_w, stride, filterBoxes, objProbs,
                                     classId, conf_threshold, app_ctx->output_attrs[i].zp, app_ctx->output_attrs[i].scale,index);
        }
        else
        {
            validCount += process_fp32((float *)_outputs[i].buf, grid_h, grid_w, stride, filterBoxes, objProbs,
                                     classId, conf_threshold, app_ctx->output_attrs[i].zp, app_ctx->output_attrs[i].scale, index);
        }
        index += grid_h * grid_w;
    }

    // no object detect
    if (validCount <= 0) {
        return 0;
    }
    int last_count = 0;
    od_results->count = 0;

    /* box valid detect target */
    for (int i = 0; i < validCount; ++i) {
        if (last_count >= OBJ_NUMB_MAX_SIZE) {
            continue;
        }

        float x1 = filterBoxes[i * 5 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[i * 5 + 1] - letter_box->y_pad;
        float w = filterBoxes[i * 5 + 2];
        float h = filterBoxes[i * 5 + 3];
        int keypoints_index = (int)filterBoxes[i * 5 + 4];

        for (int j = 0; j < 17; ++j) {
            if (app_ctx->is_quant) {
                od_results->results[last_count].keypoints[j][0] = ((float)((rknpu2::float16 *)_outputs[3].buf)[j*3*8400+0*8400+keypoints_index] 
                                                                - letter_box->x_pad)/ letter_box->scale;
                od_results->results[last_count].keypoints[j][1] = ((float)((rknpu2::float16 *)_outputs[3].buf)[j*3*8400+1*8400+keypoints_index] 
                                                                    - letter_box->y_pad)/ letter_box->scale;
                od_results->results[last_count].keypoints[j][2] = (float)((rknpu2::float16 *)_outputs[3].buf)[j*3*8400+2*8400+keypoints_index];

            }
            else
            {
                od_results->results[last_count].keypoints[j][0] = (((float *)_outputs[3].buf)[j*3*8400+0*8400+keypoints_index] 
                                                                - letter_box->x_pad)/ letter_box->scale;
                od_results->results[last_count].keypoints[j][1] = (((float *)_outputs[3].buf)[j*3*8400+1*8400+keypoints_index] 
                                                                    - letter_box->y_pad)/ letter_box->scale;
                od_results->results[last_count].keypoints[j][2] = ((float *)_outputs[3].buf)[j*3*8400+2*8400+keypoints_index];
            }
        }

        int id = classId[i];
        float obj_conf = objProbs[i];
        od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.right = (int)(clamp(x1+w, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.bottom = (int)(clamp(y1+h, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}


