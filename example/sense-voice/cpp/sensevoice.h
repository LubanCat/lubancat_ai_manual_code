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

#ifndef _RKNN_DEMO_SENSEVOICE_H_
#define _RKNN_DEMO_SENSEVOICE_H_

#include "rknn_api.h"
#include "audio_utils.h"
#include <iostream>
#include <vector>
#include <string>
#include "process.h"

typedef struct
{
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
} rknn_sensevoice_context_t;

int init_sensevoice_model(const char *model_path, rknn_sensevoice_context_t *app_ctx);
int release_sensevoice_model(rknn_sensevoice_context_t *app_ctx);
int run_sensevoice(rknn_sensevoice_context_t *app_ctx, std::vector<float> audio_data,
    int length, int language,  int text_norm, VocabEntry *vocab, std::vector<std::string> &recognized_text);

#endif //_RKNN_DEMO_SENSEVOICE_H_