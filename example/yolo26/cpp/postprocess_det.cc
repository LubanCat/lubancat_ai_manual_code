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

#include "yolo26_det.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <set>
#include <vector>

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId)
        {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static uint8_t qnt_f32_to_affine_u8(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    uint8_t res = (uint8_t)__clip(dst_val, 0, 255);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static void compute_dfl(float* tensor, int dfl_len, float* box){
    for (int b=0; b<4; b++){
        float exp_t[dfl_len];
        float exp_sum=0;
        float acc_sum=0;
        for (int i=0; i< dfl_len; i++){
            exp_t[i] = exp(tensor[i+b*dfl_len]);
            exp_sum += exp_t[i];
        }
        
        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i]/exp_sum *i;
        }
        box[b] = acc_sum;
    }
}

static int process_i8(int8_t *input, int32_t zp, float scale,
                      int grid_h, int grid_w, int stride,
                      std::vector<float> &boxes, 
                      std::vector<float> &objProbs, 
                      std::vector<int> &classId, 
                      float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, zp, scale);
    int input_loc_len =  4;

    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < OBJ_CLASS_NUM; a++) {
                if(input[(input_loc_len + a)*grid_w * grid_h + h * grid_w + w ] >= score_thres_i8) { //[1,tensor_len,grid_h,grid_w]
                    float box_conf_f32 = deqnt_affine_to_f32(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w ], zp, scale);
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = deqnt_affine_to_f32(input[i * grid_w * grid_h + h * grid_w + w], zp, scale);
                    }
                    printf("loc: %.2f, %.2f, %.2f, %.2f \n", loc[0], loc[1], loc[2], loc[3]);
                    printf("hw: %d, %d \n", h, w);
                    float x1,y1,w_,h_;
                    x1 = (w + 0.5 - loc[0])*stride;
                    y1 = (h + 0.5 - loc[1])*stride;
                    w_ = (loc[0] + loc[2])*stride;
                    h_ = (loc[1] + loc[3])*stride;
                    boxes.push_back(x1);//x
                    boxes.push_back(y1);//y
                    boxes.push_back(w_);//w
                    boxes.push_back(h_);//h
                    objProbs.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;

                    printf("x1=%.2f, y1=%.2f, w=%.2f, h=%.2f, objProb=%.2f, classId=%d\n", x1, y1, w_, h_, box_conf_f32, a);
                }
            }
        }
    }
    printf("validCount=%d\n", validCount);
    printf("grid h-%d, w-%d, stride %d\n", grid_h, grid_w, stride);
    return validCount;

}

static int process_fp32(float *input,
                        int grid_h, int grid_w, int stride,
                        std::vector<float> &boxes, 
                        std::vector<float> &objProbs, 
                        std::vector<int> &classId, 
                        float threshold)
{
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
                    objProbs.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}


#if defined(RV1106_1103)
static int process_i8_rv1106(int8_t *tensor, int32_t tensor_zp, float tensor_scale,
                             int grid_h, int grid_w, int stride,
                             std::vector<float> &boxes,
                             std::vector<float> &objProbs,
                             std::vector<int> &classId,
                             float threshold) {
    int validCount = 0;
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, tensor_zp, tensor_scale);
    int input_loc_len =  4;
    int tensor_len = input_loc_len + OBJ_CLASS_NUM;

    // 1106 NHWC
    for (int i = 0; i < grid_h; i++) {
        for (int j = 0; j < grid_w; j++) {
            for (int a = 0; a < OBJ_CLASS_NUM; a++) {
                if(tensor[tensor_len * (grid_w*i+j) + a + input_loc_len] >= score_thres_i8) {
                    float box_conf_f32 = deqnt_affine_to_f32(tensor[tensor_len * (grid_w*i+j) + a + input_loc_len], tensor_zp, tensor_scale);
                    float loc[input_loc_len];
                    for (int k = 0; k < input_loc_len; ++k) {
                        loc[k] = deqnt_affine_to_f32(tensor[tensor_len * (grid_w*i+j) + k], tensor_zp, tensor_scale);
                    }

                    float x1,y1,w_,h_;
                    x1 = (j + 0.5 - loc[0])*stride;
                    y1 = (i + 0.5 - loc[1])*stride;
                    w_ = (loc[0] + loc[2])*stride;
                    h_ = (loc[1] + loc[3])*stride;
                    boxes.push_back(x1);//x
                    boxes.push_back(y1);//y
                    boxes.push_back(w_);//w
                    boxes.push_back(h_);//h
                    objProbs.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}
#endif

int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results)
{
#if defined(RV1106_1103) 
    rknn_tensor_mem **_outputs = (rknn_tensor_mem **)outputs;
#else
    rknn_output *_outputs = (rknn_output *)outputs;
#endif
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

    // default 3 branch
    // int dfl_len = app_ctx->output_attrs[0].dims[1] /4;
    int output_per_branch = app_ctx->io_num.n_output / 3;
    for (int i = 0; i < 3; i++)
    {
#if defined(RV1106_1103)
        int box_idx = i * output_per_branch;
        grid_h = app_ctx->output_attrs[box_idx].dims[1];
        grid_w = app_ctx->output_attrs[box_idx].dims[2];
        stride = model_in_h / grid_h;
        
        if (app_ctx->is_quant) {
            validCount += process_i8_rv1106((int8_t *)_outputs[box_idx]->virt_addr, app_ctx->output_attrs[box_idx].zp, app_ctx->output_attrs[box_idx].scale,
                                grid_h, grid_w, stride, filterBoxes, objProbs, classId, conf_threshold);
        }
        else
        {
            printf("RV1106/1103 only support quantization mode\n");
            return -1;
        }

#else
        int box_idx = i*output_per_branch;

        grid_h = app_ctx->output_attrs[box_idx].dims[2];
        grid_w = app_ctx->output_attrs[box_idx].dims[3];
        stride = model_in_h / grid_h;

        if (app_ctx->is_quant)
        {
            validCount += process_i8((int8_t *)_outputs[box_idx].buf, app_ctx->output_attrs[box_idx].zp, app_ctx->output_attrs[box_idx].scale,
                                     grid_h, grid_w, stride, filterBoxes, objProbs, classId, conf_threshold);
        }
        else
        {
            validCount += process_fp32((float *)_outputs[box_idx].buf, grid_h, grid_w, stride,
                                        filterBoxes, objProbs, classId, conf_threshold);
        }
#endif

    }

    // no object detect
    if (validCount <= 0)
    {
        return 0;
    }
    printf("validCount=%d\n", validCount);

    int last_count = 0;
    od_results->count = 0;

    /* box valid detect target */
    for (int i = 0; i < validCount; ++i)
    {
        if (last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }
        float x1 = filterBoxes[i * 4 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[i * 4 + 1] - letter_box->y_pad;
        float x2 = x1 + filterBoxes[i * 4 + 2];
        float y2 = y1 + filterBoxes[i * 4 + 3];
        int id = classId[i];
        float obj_conf = objProbs[i];

        od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}

