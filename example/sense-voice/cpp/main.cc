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

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <vector>
#include <string>
#include <algorithm> 

#include "sensevoice.h"
#include "audio_utils.h"
#include "parser.h"

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    int ret;
    TIMER timer;
    float infer_time = 0.0;
    float audio_length = 0.0;
    float rtf = 0.0;

    int length;
    int language;
    int text_norm;

    rknn_sensevoice_context_t app_ctx;
    std::vector<float> input_data;
    std::vector<std::string> recognized_text;
    VocabEntry vocab[VOCAB_LEN];
    CMVNData cmvn_data;
    audio_buffer_t audio;

    Args args = parse_args(argc, argv);
    if (args.use_itn == 0)
    {
        text_norm = 15;
    } else if (args.use_itn == 1)
    {
        text_norm = 14;
    }
    else
    {
        std::cerr << "Invalid use_itn value. It should be 0 or 1." << std::endl;
        return -1;
    }

    if(args.language == "zh") {
        language = 3;
    } else if(args.language == "en") {
        language = 4;
    } else if(args.language == "ja") {
        language = 11;
    } else if(args.language == "ko") {
        language = 12;
    } else if(args.language == "auto") {
        language = 0;
    } else {
        std::cerr << "Unsupported language: " << args.language << std::endl;
        return -1;
    }

    memset(&app_ctx, 0, sizeof(rknn_sensevoice_context_t));
    memset(vocab, 0, sizeof(vocab));
    memset(&audio, 0, sizeof(audio_buffer_t));

    timer.tik();
    ret = read_audio(args.audio_path.data(), &audio);
    if (ret != 0)
    {
        printf("read audio fail! ret=%d audio_path=%s\n", ret, args.audio_path.data());
        goto out;
    }

    if (audio.num_channels == 2)
    {
        ret = convert_channels(&audio);
        if (ret != 0)
        {
            printf("convert channels fail! ret=%d\n", ret);
            goto out;
        }
    }

    if (audio.sample_rate != SAMPLE_RATE)
    {
        ret = resample_audio(&audio, audio.sample_rate, SAMPLE_RATE);
        if (ret != 0)
        {
            printf("resample audio fail! ret=%d\n", ret);
            goto out;
        }
    }
    timer.tok();
    timer.print_time("read_audio & convert_channels & resample_audio");

    timer.tik();
    ret = load_cmvn("./model/am.mvn", cmvn_data);
    if (ret != 0)
    {
        printf("load_cmvn fail! ret=%d vocab_path=%s\n", ret);
        goto out;
    }

    ret = read_vocab(args.tokens.data(), vocab);
    if (ret != 0)
    {
        printf("read vocab fail! ret=%d vocab_path=%s\n", ret, args.tokens.data());
        goto out;
    }
    timer.tok();
    timer.print_time("load_cmvn & read_vocab");

    // Initialize  model
    timer.tik();
    ret = init_sensevoice_model(args.model_path.data(), &app_ctx);
    if (ret != 0)
    {
        printf("init_sensevoice_model fail! ret=%d encoder_path=%s\n", ret, args.model_path.data());
        goto out;
    }
    timer.tok();
    timer.print_time("init_whisper_encoder_model");

    // Run inference
    timer.tik();
    audio_preprocess(&audio, length, cmvn_data, input_data);
    ret = run_sensevoice(&app_ctx, input_data, length, language, text_norm, vocab, recognized_text);
    if (ret != 0)
    {
        printf("run_sensevoice fail! ret=%d\n", ret);
        goto out;
    }
    timer.tok();
    timer.print_time("run_sensevoice");

    // print result
    std::cout << "\nOutput: ";
    for (const auto &str : recognized_text)
    {
        std::cout << str;
    }
    std::cout << std::endl;

    infer_time = timer.get_time() / 1000.0;               // sec
    audio_length = audio.num_frames / (float)SAMPLE_RATE; // sec
    audio_length = audio_length > (float)CHUNK_LENGTH ? (float)CHUNK_LENGTH : audio_length;
    rtf = infer_time / audio_length;
    printf("\nReal Time Factor (RTF): %.3f / %.3f = %.3f\n", infer_time, audio_length, rtf);

out:

    ret = release_sensevoice_model(&app_ctx);
    if (ret != 0)
    {
        printf("release_sensevoice_model encoder_context fail! ret=%d\n", ret);
    }

    for (int i = 0; i < VOCAB_LEN; ++i)
    {
        if (vocab[i].token != NULL)
        {
            free(vocab[i].token);
            vocab[i].token = NULL;
        }
    }

    if (audio.data != NULL)
    {
        free(audio.data);
    }

    return 0;
}
