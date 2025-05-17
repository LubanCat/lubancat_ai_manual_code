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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "melotts.h"
#include "parse_args.h"

#include "lexicon.hpp"
#include "split.hpp"

static std::vector<int64_t> intersperse(const std::vector<int>& lst, int item) {
    std::vector<int64_t> result(lst.size() * 2 + 1, item);
    for (size_t i = 1; i < result.size(); i+=2) {
        result[i] = lst[i / 2];
    }
    return result;
}

static std::vector<int64_t> pad_or_trim(std::vector<int64_t>& vec, int max_size) {
    if (vec.size() < max_size) {
        vec.resize(max_size, 0);
    } else if (vec.size() > max_size) {
        vec.resize(max_size);
    }
    return vec;
}

const std::map<std::string, int> language_id_map = { 
    {"ZH", 0},
    {"JP", 1},
    {"EN", 2},
    {"ZH_MIX_EN", 3},
    {"KR", 4},
    {"SP", 5},
    {"ES", 5},
    {"FR", 6}
};

int main(int argc, char **argv)
{
    int ret;
    TIMER timer;

    rknn_melotts_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_melotts_context_t));

    float infer_time = 0.0;
    float audio_length = 0.0;
    float rtf = 0.0;

    std::vector<float> output_wav_data;

    Args args = parse_args(argc, argv);
    const char *encoder_path = args.encoder_path.c_str();
    const char *decoder_path = args.decoder_path.c_str();
    // const char *bert_path = args.bert_model_path.c_str();
    const char *audio_save_path  = args.output_filename.c_str();

    // 默认ZH是ZH_MIX_EN
    if(!args.language.compare("ZH"))
    {
        args.language = "ZH_MIX_EN";
    }

    // Load lexicon
    timer.tik();
    // Lexicon lexicon(LEXICON_EN_FILE, TOKENS_EN_FILE);
    Lexicon lexicon(LEXICON_ZH_FILE, TOKENS_ZH_FILE);
    timer.tok();
    timer.print_time("Lexicon init");

    // lang_ids
    int value = 3;
    auto it = language_id_map.find(args.language);
    if (it != language_id_map.end()) {
        value = it->second;
    } else {
        std::cerr << "Language not found!" << std::endl;
        return 1;
    }

    // Split sentences
    auto sentences = split_sentence(args.input_text, 40, args.language);

    // init encoder model
    timer.tik();
    ret = init_melotts_model(encoder_path, &rknn_app_ctx.encoder_context);
    if (ret != 0)
    {
        printf("init_melotts_model fail! ret=%d encoder_path=%s\n", ret, encoder_path);
        goto out;
    }
    timer.tok();
    timer.print_time("init_melotts_encoder_model");

    // init encoder model
    timer.tik();
    ret = init_melotts_model(decoder_path, &rknn_app_ctx.decoder_context);
    if (ret != 0)
    {
        printf("init_melotts_model fail! ret=%d decoder_path=%s\n", ret, decoder_path);
        goto out;
    }
    timer.tok();
    timer.print_time("init_melotts_decoder_model");

    // init bert    
    if(!args.disable_bert) {
        // TODO init_melotts_model
        goto out;
    } else {
        std::cout << "disable bert model" << std::endl;
    }

    // inference
    timer.tik();
    for (auto& s : sentences) {
        printf("Split sentence: %s\n", s.c_str());
        std::vector<float> output_data(PREDICTED_LENGTHS_MAX*PREDICTED_BATCH); 

 	    // Convert sentence to phones and tones
        s = "_" + s + "_";
        std::vector<int> phones_bef, tones_bef;
        lexicon.convert(s, phones_bef, tones_bef);

        std::vector<int> lang_ids_bef(phones_bef.size(), value);

        // Add blank between words
        auto phones = intersperse(phones_bef, 0);
        auto tones = intersperse(tones_bef, 0);
        auto lang_ids = intersperse(lang_ids_bef, 0);

        int64_t phone_len = phones.size();

        // pad or trim
        pad_or_trim(tones, MAX_LENGTH);
        pad_or_trim(phones, MAX_LENGTH);
        pad_or_trim(lang_ids, MAX_LENGTH);

        int output_lengths = inference_melotts_model(&rknn_app_ctx, phones, phone_len, tones, lang_ids, args.speak_id, args.speed, args.disable_bert, output_data);
        if (output_lengths < 0)
        {
            printf("inference_melotts_model fail! ret=%d\n", output_lengths);
            goto out;
        }

        int actual_size = output_lengths * PREDICTED_BATCH;
        output_wav_data.insert(output_wav_data.end(), output_data.begin(), output_data.begin() + actual_size);
    }
    timer.tok();
    timer.print_time("inference ");

    infer_time = timer.get_time() / 1000.0; // sec
    std::cout << "output_wav_data size:" <<  output_wav_data.size()  <<  std::endl;
    audio_length = (float)output_wav_data.size() / SAMPLE_RATE;        // sec
    printf("audio_length: %f", audio_length);
    rtf = infer_time / audio_length;
    printf("\nReal Time Factor (RTF): %.3f / %.3f = %.3f\n", infer_time, audio_length, rtf);
    printf("\nThe output wav file is saved: %s\n", audio_save_path);

    timer.tik();
    ret = save_audio(audio_save_path, output_wav_data.data(), output_wav_data.size(), SAMPLE_RATE, 1);
    if (ret != 0)
    {
        printf("save_audio fail! ret=%d\n", ret);
        return ret;
    }
    timer.tok();
    timer.print_time("save_audio");

out:
    // release model
    ret = release_melotts_model(&rknn_app_ctx.encoder_context);
    if (ret != 0)
    {
        printf("release_mms_tts_model encoder_context fail! ret=%d\n", ret);
    }
    ret = release_melotts_model(&rknn_app_ctx.decoder_context);
    if (ret != 0)
    {
        printf("release_ppocr_model decoder_context fail! ret=%d\n", ret);
    }
    return 0;
}

