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

#include "sensevoice.h"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

#include <opencv2/opencv.hpp>

#include "kaldi-native-fbank/csrc/online-feature.h"
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/mel-computations.h"

int read_vocab(const char *fileName, VocabEntry *vocab)
{
    FILE *fp;
    char line[512];
 
    fp = fopen(fileName, "r");
    if (fp == NULL)
    {
        perror("Error opening file");
        return -1;
    }
 
    int count = 0;
    while (fgets(line, sizeof(line), fp))
    {
        char *last_space = strrchr(line, ' ');
        
        if (!last_space) {
            continue;
        }
 
        *last_space = '\0';
        vocab[count].token = strdup(line);
        vocab[count].index = atoi(last_space + 1);
        
        count++;
    }
 
    fclose(fp);
    return 0;
}

void logits_argmax(float *array, int *index)
{
    int start_index = 0;
    int end_index = 0;

    for(int j = 0; j < OUTPUT_LEN; j++)
    {
        start_index = j * VOCAB_LEN;
        end_index = (j+1) * VOCAB_LEN;
        int max_index = start_index;
        float max_value = array[start_index];
        for (int i = start_index; i < end_index; i++)
        {
            if (array[i] > max_value)
            {
                max_value = array[i];
                max_index = i;
            }
        }
        index[j] = max_index - start_index;
    }
}

int load_cmvn(const std::string& cmvn_file, CMVNData& cmvn_data) {
    std::ifstream file(cmvn_file);
    if (!file.is_open()) {
        return -1;    // 打开CMVN文件失败
    }

    bool found_first = false;
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("<LearnRateCoef>") != 0) {
            continue;
        }
        
        // 分割行
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;
        while (iss >> token) {
            tokens.push_back(token);
        }
        
        if (tokens.size() < 5) {
            continue;
        }
        
        // 提取第4个字段到倒数第二个字段 (索引3到size-2)
        std::vector<float> values;
        for (size_t i = 3; i < tokens.size() - 1; ++i) {
            try {
                values.push_back(std::stof(tokens[i]));
            } catch (...) {
                return -2; // 无效的浮点数值
            }
        }
        
        // 第一个<LearnRateCoef>行包含neg_mean
        if (!found_first) {
            cmvn_data.means = std::move(values);
            found_first = true;
        } 
        // 第二个<LearnRateCoef>行包含inv_stddev
        else {
            cmvn_data.vars = std::move(values);
            break;
        }
    }
    
    // 验证
    if (!found_first || cmvn_data.vars.empty()) {
        return -3; 
    }
    
    if (cmvn_data.means.size() != cmvn_data.vars.size()) {
        return -4;
    }
    
    return 0;
}

void apply_cmvn(float* features, int num_frames, int num_features, const CMVNData& cmvn) {
    if (cmvn.means.size() != static_cast<size_t>(num_features) || 
        cmvn.vars.size() != static_cast<size_t>(num_features)) {
        std::cout << "CMVN dimensions do not match feature dimensions!" << std::endl;
    }

    for (int i = 0; i < num_frames; ++i) {
        for (int j = 0; j < num_features; ++j) {
            features[i * num_features + j] = 
                (features[i * num_features + j] + cmvn.means[j]) * cmvn.vars[j];
        }
    }
}

static std::vector<float> apply_lfr(const std::vector<float>& inputs, int lfr_m, int lfr_n) 
{
    int frame_dim = N_MELS;

    int T = inputs.size() / frame_dim; // 总帧数

    int T_lfr = static_cast<int>(ceil(T / static_cast<double>(lfr_n)));
    int pad_size = (lfr_m - 1) / 2;

    int total_len = (T + pad_size) * frame_dim;
    std::vector<float> padded_inputs(total_len);

    int first_frame_offset = 0;
    for (int i = 0; i < pad_size; ++i) {
        memcpy(&padded_inputs[i * frame_dim], &inputs[0], frame_dim * sizeof(float));
        first_frame_offset = i * frame_dim;
    }
    memcpy(&padded_inputs[pad_size * frame_dim], inputs.data(), inputs.size() * sizeof(float));

    T += pad_size;
    std::vector<float> LFR_outputs;
    LFR_outputs.reserve(T_lfr * lfr_m * frame_dim);

    for (int i = 0; i < T_lfr; ++i) {
        int start_idx = i * lfr_n * frame_dim;
        int end_idx = start_idx + lfr_m * frame_dim;

        if (end_idx <= padded_inputs.size()) {
            LFR_outputs.insert(LFR_outputs.end(), 
                                &padded_inputs[start_idx], 
                                &padded_inputs[end_idx]);
        } else {
            int remaining = padded_inputs.size() - start_idx;
            LFR_outputs.insert(LFR_outputs.end(),
                                &padded_inputs[start_idx], 
                                &padded_inputs[start_idx + remaining]);

            int pad_num = lfr_m * frame_dim - remaining;
            int last_frame_start = (T - 1) * frame_dim;
            for (int p = 0; p < pad_num; ++p) {
                int idx = last_frame_start + (p % frame_dim);
                LFR_outputs.push_back(padded_inputs[idx]);
            }
        }
    }

    return LFR_outputs;
}

void audio_preprocess(audio_buffer_t *audio, int &len, CMVNData &cmvn_data, std::vector<float> &features)
{
    knf::FbankOptions opts;
    opts.frame_opts.dither = 0;
    opts.frame_opts.samp_freq = SAMPLE_RATE;
    opts.frame_opts.window_type = "hanning";
    opts.frame_opts.snip_edges = false;
    opts.mel_opts.num_bins = N_MELS;

    int audio_length = audio->num_frames;
    std::vector<float> ori_audio_data(audio->data, audio->data + audio_length);

    knf::OnlineFbank fbank(opts);
    for (int32_t i = 0; i < audio_length; ++i) {
        float s = ori_audio_data[i] * 32768;
        fbank.AcceptWaveform(SAMPLE_RATE, &s, 1);
    }

    std::vector<float> mel_data;

    int32_t n = fbank.NumFramesReady();
    for (int32_t i = 0; i != n; ++i) {
        const float *frame = fbank.GetFrame(i);
        for (int32_t k = 0; k != N_MELS; ++k) {
            mel_data.push_back(frame[k]);
        }
    }

    // Apply LFR
    features = apply_lfr(mel_data, LFR_M, LFR_N);

    // Apply CMVN
    int num_features = FEATURES_LEN;
    len = features.size() / num_features;
    apply_cmvn(features.data(), len, num_features, cmvn_data);

    // pad or trim
    if (len < INPUT_LENGTH)
    {
        features.resize(INPUT_LENGTH * FEATURES_LEN, 0.0f); // Pad with zeros
    }
    else if (len > INPUT_LENGTH)
    {
        features.resize(INPUT_LENGTH * FEATURES_LEN); // Trim
        std::cout << "features trim to INPUT_LENGTH!" << std::endl;
    }
}