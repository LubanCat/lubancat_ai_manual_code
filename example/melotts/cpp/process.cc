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

// Modify from https://github.com/airockchip/rknn_model_zoo/blob/main/examples/mms_tts/cpp/process.cc

#include "melotts.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <map>
#include <string.h>
#include <algorithm>
#include <numeric>

static void compute_output_padding_mask(std::vector<float> &output_padding_mask, int predicted_lengths_max_real, int predicted_lengths_max)
{
    int index = 0;
    std::transform(output_padding_mask.begin(), output_padding_mask.end(), output_padding_mask.begin(),
                   [&index, predicted_lengths_max_real](int i)
                   {
                       float result = (float)(index < predicted_lengths_max_real) ;
                       ++index;
                       return result;
                   });
}

static void compute_attn_mask(std::vector<float> &output_padding_mask, std::vector<float> &input_padding_mask,
                              std::vector<int> &attn_mask, int predicted_lengths_max, int input_padding_mask_size)
{
    int index = 0;
    std::transform(attn_mask.begin(), attn_mask.end(), attn_mask.begin(),
                    [&index, &output_padding_mask, &input_padding_mask, predicted_lengths_max, input_padding_mask_size](int k)
                    {
                        int i = index / predicted_lengths_max;
                        int j = index % predicted_lengths_max;
                        ++index;
                        return int(output_padding_mask[j] * input_padding_mask[i]);
                    });
}

static void compute_duration(const std::vector<float> &exp_log_duration, const std::vector<float> &input_padding_mask,
                             std::vector<float> &duration, float length_scale)
{
    std::transform(exp_log_duration.begin(), exp_log_duration.end(), input_padding_mask.begin(), duration.begin(),
                   [length_scale](float exp_log_val, float mask_val)
                   {
                       return ceil(exp_log_val * mask_val * length_scale);
                   });
}

static void compute_valid_indices(const std::vector<float> &cum_duration, std::vector<int> &valid_indices, int input_padding_mask_size, int predicted_lengths_max)
{
    std::vector<int> indices(valid_indices.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::for_each(indices.begin(), indices.end(),
                  [cum_duration, &valid_indices, predicted_lengths_max](int index)
                  {
                      int i = index / predicted_lengths_max;
                      int j = index % predicted_lengths_max;
                      valid_indices[index] = j < cum_duration[i] ? 1 : 0;
                  });
}

static std::vector<float> exp_vector(const std::vector<float> &vec)
{
    std::vector<float> result(vec.size());
    std::transform(vec.begin(), vec.end(), result.begin(), [](float v)
                   { return exp(v); });
    return result;
}

static std::vector<float> cumsum(const std::vector<float> &vec)
{
    std::vector<float> result(vec.size());
    std::partial_sum(vec.begin(), vec.end(), result.begin());
    return result;
}

static void transpose_mul(const std::vector<int> &input, int input_rows, int input_cols, std::vector<int> attn_mask, std::vector<float> &output)
{
    std::vector<int> indices(input.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(indices.begin(), indices.end(),
                  [&input, &attn_mask, &output, input_rows, input_cols](int index)
                  {
                    int i = index / input_cols;
                    int j = index % input_cols;
                    output[j * input_rows + i] = (float)(input[index] * attn_mask[index]);
                  });
}

static void compute_pad_indices(const std::vector<int> &valid_indices, std::vector<int> &sliced_indices, int input_length, int output_length)
{
    int padded_length = input_length + 1;
    std::vector<int> padded_indices(padded_length * output_length, 0);

    std::copy(valid_indices.begin(), valid_indices.end(), padded_indices.begin() + output_length);

    std::copy(padded_indices.begin(), padded_indices.begin() + input_length * output_length, sliced_indices.begin());

    std::transform(valid_indices.begin(), valid_indices.end(), sliced_indices.begin(),
                   sliced_indices.begin(), std::minus<int>());
}

void middle_process(std::vector<float> log_w, std::vector<float> x_mask, std::vector<float> &attn,
                    std::vector<float> &y_mask, float speed, int &predicted_lengths_max_real)
{
    float length_scale = 1.0f / speed;
    std::vector<float> w(LOG_DURATION_SIZE);

    std::vector<float> exp_log_w = exp_vector(log_w);
    compute_duration(exp_log_w, x_mask, w, length_scale);
    float predicted_length_sum = std::accumulate(w.begin(), w.end(), 0.0f);
    predicted_lengths_max_real = std::max(1.0f, predicted_length_sum);
    int predicted_lengths_max = PREDICTED_LENGTHS_MAX;
    if(predicted_lengths_max_real > predicted_lengths_max)
    {
        //TODO
        printf("predicted_lengths_max_real > PREDICTED_LENGTHS_MAX \n");
        predicted_lengths_max_real = predicted_lengths_max;
    }
    compute_output_padding_mask(y_mask, predicted_lengths_max_real, predicted_lengths_max);
    int x_mask_size = MAX_LENGTH;
    std::vector<int> attn_mask(predicted_lengths_max * x_mask_size);
    compute_attn_mask(y_mask, x_mask, attn_mask, predicted_lengths_max, x_mask_size);
    std::vector<float> cum_duration = cumsum(w);
    std::vector<int> valid_indices(x_mask_size * predicted_lengths_max, 0);
    compute_valid_indices(cum_duration, valid_indices, x_mask_size, predicted_lengths_max);
    std::vector<int> padded_indices(x_mask_size * predicted_lengths_max, 0);
    compute_pad_indices(valid_indices, padded_indices, x_mask_size, predicted_lengths_max);
    transpose_mul(padded_indices, x_mask_size, predicted_lengths_max, attn_mask, attn);
}