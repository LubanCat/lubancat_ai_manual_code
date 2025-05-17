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
#include <math.h>
#include <vector>

#include "process.h"
#include "melotts.h"
#include "file_utils.h"

static void dump_input_dynamic_range(rknn_input_range *dyn_range)
{
    std::string range_str = "";
    for (int n = 0; n < dyn_range->shape_number; ++n)
    {
        range_str += n == 0 ? "[" : ",[";
        range_str += dyn_range->n_dims < 1 ? "" : std::to_string(dyn_range->dyn_range[n][0]);
        for (int i = 1; i < dyn_range->n_dims; ++i)
        {
            range_str += ", " + std::to_string(dyn_range->dyn_range[n][i]);
        }
        range_str += "]";
    }

    printf("  index=%d, name=%s, shape_number=%d, range=[%s], fmt = %s\n", dyn_range->index, dyn_range->name,
           dyn_range->shape_number, range_str.c_str(), get_format_string(dyn_range->fmt));
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    char dims_str[100];
    char temp_str[100];
    memset(dims_str, 0, sizeof(dims_str));
    for (int i = 0; i < attr->n_dims; i++)
    {
        strcpy(temp_str, dims_str);
        if (i == attr->n_dims - 1)
        {
            sprintf(dims_str, "%s%d", temp_str, attr->dims[i]);
        }
        else
        {
            sprintf(dims_str, "%s%d, ", temp_str, attr->dims[i]);
        }
    }

    printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, dims_str, attr->n_elems, attr->size, get_format_string(attr->fmt),
           get_type_string(attr->type), get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

int init_melotts_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret;
    int model_len = 0;
    rknn_context ctx = 0;

    ret = rknn_init(&ctx, (char *)model_path, model_len, 0, NULL);
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
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    return 0;
}

int release_melotts_model(rknn_app_context_t *app_ctx)
{
    if (app_ctx->input_attrs != NULL)
    {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL)
    {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }

    if (app_ctx->shape_range != NULL)
    {
        free(app_ctx->shape_range);
        app_ctx->shape_range = NULL;
    }
    if (app_ctx->rknn_ctx != 0)
    {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_encoder_model(rknn_app_context_t *app_ctx, std::vector<int64_t> &x,
    int64_t x_lengths, int64_t speaker_id, std::vector<int64_t> &tones, std::vector<int64_t> &lang_ids,
    std::vector<float> &ja_bert, std::vector<float> &logw, std::vector<float> &x_mask,
    std::vector<float> &g, std::vector<float> &m_p, std::vector<float> &logs_p)
{
    int ret;
    int n_input = 8;
    int n_output = 5;
    rknn_input inputs[n_input];
    rknn_output outputs[n_output];

    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    float sdp_ratio = SDP_RATIO;
    float noise_scale_w = NOISE_SCALE_W;

    // Set Input Data
    //['x', 'x_lengths', 'sid', 'tone', 'lang_ids', 'ja_bert', 'noise_scale_w', 'sdp_ratio']
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_INT64;
    inputs[0].fmt = RKNN_TENSOR_UNDEFINED;
    inputs[0].size = INPUT_SIZE * sizeof(int64_t);
    inputs[0].buf = (int64_t *)malloc(inputs[0].size);
    memcpy(inputs[0].buf, x.data(), inputs[0].size);

    inputs[1].index = 1;
    inputs[1].type = RKNN_TENSOR_INT64;
    inputs[1].size = 1 * sizeof(int64_t);
    inputs[1].fmt = RKNN_TENSOR_UNDEFINED;
    inputs[1].buf = (int64_t *)malloc(inputs[1].size);
    memcpy(inputs[1].buf, &x_lengths, inputs[1].size);

    inputs[2].index = 2;
    inputs[2].type = RKNN_TENSOR_INT64;
    inputs[2].size = 1 * sizeof(int64_t);
    inputs[2].fmt = RKNN_TENSOR_UNDEFINED;
    inputs[2].buf = (int64_t *)malloc(inputs[2].size);
    memcpy(inputs[2].buf, &speaker_id, inputs[2].size);

    inputs[3].index = 3;
    inputs[3].type = RKNN_TENSOR_INT64;
    inputs[3].size = INPUT_SIZE  * sizeof(int64_t);
    inputs[3].fmt = RKNN_TENSOR_UNDEFINED;
    inputs[3].buf = (int64_t *)malloc(inputs[3].size);
    memcpy(inputs[3].buf, tones.data(), inputs[3].size);

    inputs[4].index = 4;
    inputs[4].type = RKNN_TENSOR_INT64;
    inputs[4].size = INPUT_SIZE * sizeof(int64_t);
    inputs[4].fmt = RKNN_TENSOR_UNDEFINED;
    inputs[4].buf = (int64_t *)malloc(inputs[4].size);
    memcpy(inputs[4].buf, lang_ids.data(), inputs[4].size);

    inputs[5].index = 5;
    inputs[5].type = RKNN_TENSOR_FLOAT32;
    inputs[5].size = 1 * 768 * 256 * sizeof(float);
    inputs[5].buf = (float *)malloc(inputs[5].size);
    memcpy(inputs[5].buf, ja_bert.data(), inputs[5].size);

    inputs[6].index = 6;
    inputs[6].type = RKNN_TENSOR_FLOAT32;
    inputs[6].size = 1  * sizeof(float);
    inputs[6].buf = (float *)malloc(inputs[6].size);
    memcpy(inputs[6].buf, &noise_scale_w, inputs[6].size);

    inputs[7].index = 7;
    inputs[7].type = RKNN_TENSOR_FLOAT32;
    inputs[7].size = 1  * sizeof(float);
    inputs[7].buf = (float *)malloc(inputs[7].size);
    memcpy(inputs[7].buf, &sdp_ratio, inputs[7].size);

    ret = rknn_inputs_set(app_ctx->rknn_ctx, n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        goto out;
    }

    // Run
    ret = rknn_run(app_ctx->rknn_ctx, NULL);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        goto out;
    }

    // Get Output
    //["logw", "x_mask", "g", "m_p", "logs_p"]
    for (int i = 0; i < n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = 1;
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        goto out;
    }

    memcpy(logw.data(), (float *)outputs[0].buf, LOGW_SIZE * sizeof(float));
    memcpy(x_mask.data(), (float *)outputs[1].buf, X_MASK_SIZE * sizeof(float));
    memcpy(g.data(), (float *)outputs[2].buf, G_SIZE * sizeof(float));
    memcpy(m_p.data(), (float *)outputs[3].buf, M_P_SIZE * sizeof(float));
    memcpy(logs_p.data(), (float *)outputs[4].buf, LOGS_P_SIZE * sizeof(float));

out:
    // Remeber to release rknn output
    rknn_outputs_release(app_ctx->rknn_ctx, n_output, outputs);
    for (int i = 0; i < n_input; i++)
    {
        if (inputs[i].buf != NULL)
        {
            free(inputs[i].buf);
        }
    }

    return ret;
}


int inference_decoder_model(rknn_app_context_t *app_ctx, std::vector<float> &attn, std::vector<float> &y_mask, std::vector<float> &g, 
    std::vector<float> &m_p, std::vector<float> &logs_p, int &predicted_lengths_max_real, std::vector<float> &output_wav_data)
{
    int ret;
    int n_input = 6;
    int n_output = 1;
    float noise_scale = NOISE_SCALE;

    rknn_input inputs[n_input];
    rknn_output outputs[n_output];
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Set Input Data
    // ["attn", "y_mask", "g", "m_p", "logs_p", "noise_scale"],
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].size = ATTN_SIZE * sizeof(float);
    inputs[0].buf = (float *)malloc(inputs[0].size);
    memcpy(inputs[0].buf, attn.data(), inputs[0].size);

    inputs[1].index = 1;
    inputs[1].type = RKNN_TENSOR_FLOAT32;
    inputs[1].size = Y_MASK_SIZE * sizeof(float);
    inputs[1].buf = (float *)malloc(inputs[1].size);
    memcpy(inputs[1].buf, y_mask.data(), inputs[1].size);

    inputs[2].index = 1;
    inputs[2].type = RKNN_TENSOR_FLOAT32;
    inputs[2].size = G_SIZE * sizeof(float);
    inputs[2].buf = (float *)malloc(inputs[2].size);
    memcpy(inputs[2].buf, g.data(), inputs[2].size);

    inputs[3].index = 3;
    inputs[3].type = RKNN_TENSOR_FLOAT32;
    inputs[3].size = M_P_SIZE * sizeof(float);
    inputs[3].buf = (float *)malloc(inputs[3].size);
    memcpy(inputs[3].buf, m_p.data(), inputs[3].size);

    inputs[4].index = 4;
    inputs[4].type = RKNN_TENSOR_FLOAT32;
    inputs[4].size = LOGS_P_SIZE * sizeof(float);
    inputs[4].buf = (float *)malloc(inputs[4].size);
    memcpy(inputs[4].buf, logs_p.data(), inputs[4].size);

    inputs[5].index = 5;
    inputs[5].type = RKNN_TENSOR_FLOAT32;
    inputs[5].size = 1 * sizeof(float);
    inputs[5].buf = (float *)malloc(inputs[5].size);
    memcpy(inputs[5].buf, &noise_scale, inputs[5].size);

    ret = rknn_inputs_set(app_ctx->rknn_ctx, n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        goto out;
    }

    // Run
    // std::cout << "inference_decoder_model rknn_run : " << std::endl;
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        goto out;
    }

    // Get Output
    // ['y']
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(app_ctx->rknn_ctx, n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        goto out;
    }

    memcpy(output_wav_data.data(), (float *)outputs[0].buf, predicted_lengths_max_real * PREDICTED_BATCH * sizeof(float));

out:
    // Remeber to release rknn output
    rknn_outputs_release(app_ctx->rknn_ctx, n_output, outputs);
    for (int i = 0; i < n_input; i++)
    {
        if (inputs[i].buf != NULL)
        {
            free(inputs[i].buf);
        }
    }

    return ret;
}

int inference_melotts_model(rknn_melotts_context_t *app_ctx, std::vector<int64_t> &phones,
    int64_t phone_len, std::vector<int64_t> &tones, std::vector<int64_t> &lang_ids,
    int64_t speaker_id, float speed, bool disable_bert, std::vector<float> &output_wav_data)
{
    int ret;
    TIMER timer;
    std::vector<float> logw(LOGW_SIZE);
    std::vector<float> m_p(M_P_SIZE);
    std::vector<float> logs_p(LOGS_P_SIZE);
    std::vector<float> x_mask(X_MASK_SIZE);
    std::vector<float> g(G_SIZE);

    // std::vector<float> bert(1*1024*256, 0.0f);
    std::vector<float> ja_bert;
    if(disable_bert)
    {
        ja_bert.resize(1*768*256, 0.0f);
    }
    else
    {
        //TODO
        return -1;
    }

    // encoder
    timer.tik();
    ret = inference_encoder_model(&app_ctx->encoder_context, phones, phone_len, speaker_id, tones, lang_ids, ja_bert, logw,  x_mask, g, m_p, logs_p);
    if (ret != 0)
    {
        printf("inference_encoder_model fail! ret=%d\n", ret);
        return ret;
    }
    timer.tok();
    timer.print_time("inference_encoder_model");

    // middle
    timer.tik();
    int predicted_lengths_max_real = 0;
    std::vector<float> y_mask(Y_MASK_SIZE, 0.0f);
    std::vector<float> attn(ATTN_SIZE, 0.0f);
    middle_process(logw, x_mask, attn, y_mask, speed, predicted_lengths_max_real);
    timer.tok();
    timer.print_time("middle_process");

    // decoder
    timer.tik();
    ret = inference_decoder_model(&app_ctx->decoder_context, attn, y_mask, g, m_p, logs_p, predicted_lengths_max_real, output_wav_data);
    if (ret != 0)
    {
        printf("inference_decoder_model fail! ret=%d\n", ret);
        return ret;
    }
    timer.tok();
    timer.print_time("inference_decoder_model");

    return predicted_lengths_max_real;
}

