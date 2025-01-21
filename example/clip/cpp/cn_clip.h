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


#ifndef _RKNN_DEMO_CN_CLIP_H_
#define _RKNN_DEMO_CN_CLIP_H_

#include "rknn_api.h"
#include "common.h"
#include "cn_clip_tokenizer.h"
#include "rknn_clip_utils.h"

#define MAX_TEXT_NUM 12

typedef struct {
    rknn_clip_context img;
    rknn_clip_context text;
    ChineseCLIPTokenizer* cn_clip_tokenize;

    int input_img_num;
    int input_text_num;
} rknn_app_context_t;

#include "postprocess.h"

int init_cn_clip_model(const char* img_model_path,
                    const char* text_model_path,
                    rknn_app_context_t* app_ctx);

int release_cn_clip_model(rknn_app_context_t* app_ctx);

int inference_cn_clip_model(rknn_app_context_t* app_ctx,
                        image_buffer_t* img,
                        char** input_texts,
                        int text_num,
                        clip_res* out_res
                        );

#endif //_RKNN_DEMO_CN_CLIP_H_