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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>

#include <iostream>

#include "owlvit.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"
#include "clip_tokenizer.h"

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; ++i) {
        shape_str += ", " + std::to_string(attr->dims[i]);
    }

    printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size,get_format_string(attr->fmt),
           get_type_string(attr->type), get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static int init_owlvit_text_model(rknn_owlvit_context_t* app_ctx, const char* text_model_path)
{
    int ret;
    rknn_context ctx = 0;

    // Load RKNN Model
    ret = rknn_init(&ctx, (char*)text_model_path, 0, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->owlvit_text_ctx = ctx;
    app_ctx->owlvit_text_io_num = io_num;
    app_ctx->owlvit_text_input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->owlvit_text_input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->owlvit_text_output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->owlvit_text_output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    return 0;
}

static int init_owlvit_image_model(rknn_owlvit_context_t* app_ctx, const char* image_model_path)
{
    int ret;
    rknn_context ctx = 0;

    // Load RKNN Model
    ret = rknn_init(&ctx, (char*)image_model_path, 0, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->owlvit_image_ctx = ctx;

    // TODO
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type == RKNN_TENSOR_INT8)
    {
        app_ctx->is_quant = true;
    }
    else
    {
        app_ctx->is_quant = false;
    }

    app_ctx->owlvit_image_io_num = io_num;
    app_ctx->owlvit_image_input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->owlvit_image_input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->owlvit_image_output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->owlvit_image_output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

int init_owlvit_model(rknn_owlvit_context_t* app_ctx, const char* text_model_path, const char* image_model_path)
{
    int ret;

    ret = init_owlvit_text_model(app_ctx, text_model_path);
    if (ret != 0)
    {
        printf("init owlvit_text model fail! ret=%d\n", ret);
        return -1;
    }

    ret = init_owlvit_image_model(app_ctx, image_model_path);
    if (ret != 0)
    {
        printf("init owlvit_image model fail! ret=%d\n", ret);
        return -1;
    }

    return 0;
}

int release_owlvit_model(rknn_owlvit_context_t* app_ctx)
{
    // owlvit_text
    if (app_ctx->owlvit_text_input_attrs != NULL)
    {
        free(app_ctx->owlvit_text_input_attrs);
        app_ctx->owlvit_text_input_attrs = NULL;
    }
    if (app_ctx->owlvit_text_output_attrs != NULL)
    {
        free(app_ctx->owlvit_text_output_attrs);
        app_ctx->owlvit_text_output_attrs = NULL;
    }
    if (app_ctx->owlvit_text_ctx != 0)
    {
        rknn_destroy(app_ctx->owlvit_text_ctx);
        app_ctx->owlvit_text_ctx = 0;
    }

    // owlvit_image
    if (app_ctx->owlvit_image_input_attrs != NULL)
    {
        free(app_ctx->owlvit_image_input_attrs);
        app_ctx->owlvit_image_input_attrs = NULL;
    }
    if (app_ctx->owlvit_image_output_attrs != NULL)
    {
        free(app_ctx->owlvit_image_output_attrs);
        app_ctx->owlvit_image_output_attrs = NULL;
    }
    if (app_ctx->owlvit_image_ctx != 0)
    {
        rknn_destroy(app_ctx->owlvit_image_ctx);
        app_ctx->owlvit_image_ctx = 0;
    }

    return 0;
}

static int inference_owlvit_text_model(rknn_owlvit_context_t* app_ctx, int64_t* input_ids, int64_t* attention_mask, 
            int text_nums, float* image_features, float* pred_boxes, letterbox_t* letter_box, object_detect_result_list* od_results)
{
    int ret;
    rknn_input inputs[app_ctx->owlvit_text_io_num.n_input];
    rknn_output outputs[app_ctx->owlvit_text_io_num.n_output];

    int input0_size = app_ctx->owlvit_text_input_attrs[0].dims[0] *  app_ctx->owlvit_text_input_attrs[0].dims[1] *
                app_ctx->owlvit_text_input_attrs[0].dims[2] * app_ctx->owlvit_text_input_attrs[0].dims[3] * sizeof(float);
    int logits_size = app_ctx->owlvit_text_output_attrs[0].dims[0] * app_ctx->owlvit_text_output_attrs[0].dims[1] * app_ctx->owlvit_text_output_attrs[0].dims[2];
    float threshold_unsigmoid = unsigmoid(BOX_THRESH);
    int count = 0;

    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = input0_size;
    inputs[0].buf = image_features;

    inputs[1].index = 1;
    inputs[1].type = RKNN_TENSOR_INT64;
    inputs[1].fmt = RKNN_TENSOR_UNDEFINED;
    inputs[1].size = app_ctx->owlvit_text_input_attrs[1].dims[0] * app_ctx->owlvit_text_input_attrs[1].dims[1] * sizeof(int64_t);

    inputs[2].index = 2;
    inputs[2].type = RKNN_TENSOR_INT64;
    inputs[2].fmt = RKNN_TENSOR_UNDEFINED;
    inputs[2].size = app_ctx->owlvit_text_input_attrs[2].dims[0] * app_ctx->owlvit_text_input_attrs[2].dims[1] * sizeof(int64_t);

    outputs[0].index = 0;
    outputs[0].want_float = 1;

    for (int i = 0; i < text_nums; i++)
    {

        inputs[1].buf = input_ids + (i*app_ctx->owlvit_text_input_attrs[1].dims[1]);

        inputs[2].buf = attention_mask + (i*app_ctx->owlvit_text_input_attrs[2].dims[1]);

        ret = rknn_inputs_set(app_ctx->owlvit_text_ctx, app_ctx->owlvit_text_io_num.n_input, inputs);
        if (ret < 0)
        {
            printf("rknn_input_set fail! ret=%d\n", ret);
            return -1;
        }

        // Run
        ret = rknn_run(app_ctx->owlvit_text_ctx, NULL);
        if (ret < 0)
        {
            printf("rknn_run fail! ret=%d\n", ret);
            return -1;
        }

        // Get Output
        ret = rknn_outputs_get(app_ctx->owlvit_text_ctx, 1, outputs, NULL);
        if (ret < 0)
        {
            printf("rknn_outputs_get fail! ret=%d\n", ret);
            return -1;
        }

        float *logits = (float *)outputs[0].buf;

        for (size_t j = 0; j < logits_size; j++)
        {
            if (count >= OBJ_NUMB_MAX_SIZE)
            {
                break;
            }

            if (logits[j] > threshold_unsigmoid)
            {
                float xc = pred_boxes[j * 4 + 0] * app_ctx->model_width;
                float yc = pred_boxes[j * 4 + 1] * app_ctx->model_height;
                float w = pred_boxes[j * 4 + 2] * app_ctx->model_width;
                float h = pred_boxes[j * 4 + 3] * app_ctx->model_height;

                float x0 = xc - w / 2;
                float y0 = yc - h / 2;
                float x1 = xc + w / 2;
                float y1 = yc + h / 2;

                od_results->results[count].box.left  = (int)(clamp(x0 - letter_box->x_pad, 0, app_ctx->model_width) / letter_box->scale);
                od_results->results[count].box.top  = (int)(clamp(y0 - letter_box->y_pad, 0, app_ctx->model_height) / letter_box->scale);
                od_results->results[count].box.right  = (int)(clamp(x1 - letter_box->x_pad, 0, app_ctx->model_width) / letter_box->scale);
                od_results->results[count].box.bottom  = (int)(clamp(y1 - letter_box->y_pad, 0, app_ctx->model_height) / letter_box->scale);

                od_results->results[count].text_id = i;

                od_results->results[count].prop = sigmoid(logits[j]);
                count++;
            }
        }

        // release rknn output
        rknn_outputs_release(app_ctx->owlvit_text_ctx, 1, outputs);
    }

    od_results->count = count;

    return 0;
}

static int inference_owlvit_image_model(rknn_owlvit_context_t* app_ctx, image_buffer_t* img, float *pred_boxes, float *image_features)
{
    int ret;
    rknn_input inputs[app_ctx->owlvit_image_io_num.n_input];
    rknn_output outputs[app_ctx->owlvit_image_io_num.n_output];
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    inputs[0].buf = img->virt_addr;

    ret = rknn_inputs_set(app_ctx->owlvit_image_ctx, app_ctx->owlvit_image_io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    printf("rknn_run owlvit_image_ctx\n");
    ret = rknn_run(app_ctx->owlvit_image_ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Get Output
    outputs[0].index = 0;
    outputs[0].want_float = 1;

    outputs[1].index = 1;
    outputs[1].want_float = (!app_ctx->is_quant);

    ret = rknn_outputs_get(app_ctx->owlvit_image_ctx, app_ctx->owlvit_image_io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }

    // CHW --> HWC
    float *output0 = (float *)outputs[0].buf;
    for (int h = 0; h < 24; h++) {
        for(int w = 0; w < 768; w++) {
            for(int c = 0; c < 24; c++){
                int input_index =  c * 24 * 768 + h * 768 + w;
                int output_index =  h * 768 * 24 + w * 24 + c;
                image_features[output_index] = output0[input_index];
            }
        }
    }

    memcpy(pred_boxes, (float*)outputs[1].buf, CNT_PRED_BOXES * 4 * sizeof(float));

    // Remeber to release rknn output
    rknn_outputs_release(app_ctx->owlvit_image_ctx, app_ctx->owlvit_image_io_num.n_output, outputs);

    return 0;
}

int inference_owlvit_model(rknn_owlvit_context_t* app_ctx, image_buffer_t* img, char** text_input, int text_nums, object_detect_result_list* od_results)
{
    int ret;
    int bg_color = 114;

    memset(od_results, 0x00, sizeof(*od_results));

    // image pre process
    image_buffer_t dst_img;
    letterbox_t letter_box;
    float *pred_boxes = (float*)malloc(CNT_PRED_BOXES * 4 * sizeof(float));
    float *image_features = (float*)malloc(LEN_IMAGE_FEATURE * sizeof(float));

    dst_img.width = app_ctx->model_width;
    dst_img.height = app_ctx->model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char *)malloc(dst_img.size);
    if (dst_img.virt_addr == NULL)
    {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        return -1;
    }

    // letterbox
    ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);
    if (ret < 0)
    {
        printf("convert_image_with_letterbox fail! ret=%d\n", ret);
        return -1;
    }

    // image inference
    inference_owlvit_image_model(app_ctx, &dst_img, pred_boxes, image_features);

    // text preprocess
    CLIPTokenizer* clip_tokenize = new CLIPTokenizer();

    // int sequence_len = app_ctx->owlvit_text_input_attrs[1].dims[1];  // LEN_TEXT_TOKEN
    int tokens_num = text_nums * LEN_TEXT_TOKEN;
    int64_t* tokens = (int64_t*)malloc(tokens_num * sizeof(int64_t));
    int64_t* attention_mask = (int64_t*)malloc(tokens_num * sizeof(int64_t));

    for (int i = 0; i < text_nums; i++)
    {
        std::vector<int> token = clip_tokenize->tokenize(text_input[i], LEN_TEXT_TOKEN, false);
        for (int j = 0; j < token.size(); j++)
        {
            tokens[i*LEN_TEXT_TOKEN+j] = token[j];
            attention_mask[i*LEN_TEXT_TOKEN+j] = 1;
        }

        for (size_t j = token.size(); j < LEN_TEXT_TOKEN; j++)
        {
            tokens[i*LEN_TEXT_TOKEN+j] = 0;
            attention_mask[i*LEN_TEXT_TOKEN+j] = 0;
        }
    }

    delete clip_tokenize;

    // text inference and od_results
    ret = inference_owlvit_text_model(app_ctx, tokens, attention_mask, text_nums,
                                image_features, pred_boxes, &letter_box, od_results);
    if (ret != 0)
    {
        printf("inference_owlvit_text_model fail! ret=%d\n", ret);
        return -1;
    }

    // relese
    if (tokens != NULL)
    {
        free(tokens);
    }

    if (attention_mask != NULL)
    {
        free(attention_mask);
    }

    if (pred_boxes != NULL)
    {
        free(pred_boxes);
    }

    if (image_features != NULL)
    {
        free(image_features);
    }
    
    if (dst_img.virt_addr != NULL)
    {
        free(dst_img.virt_addr);
    }

    return 0;
}
