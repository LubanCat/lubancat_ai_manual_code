#ifndef _TTS_DEMO_PROCESS_H_
#define _TTS_DEMO_PROCESS_H_

#include "rknn_api.h"
#include "easy_timer.h"
#include "audio_utils.h"

#define SAMPLE_RATE 44100
#define MAX_LENGTH 256

#define PREDICTED_LENGTHS_MAX MAX_LENGTH*2
#define PREDICTED_BATCH 512

#define INPUT_SIZE 1 * MAX_LENGTH
#define LOGW_SIZE 1 * 1 * MAX_LENGTH
#define X_MASK_SIZE 1 * 1 * MAX_LENGTH
#define G_SIZE 1 * MAX_LENGTH * 1
#define M_P_SIZE 1 *192 * MAX_LENGTH
#define LOGS_P_SIZE 1 *192 * MAX_LENGTH
#define LOG_DURATION_SIZE 1 * 1 * MAX_LENGTH
#define ATTN_SIZE 1 * 1 * PREDICTED_LENGTHS_MAX *MAX_LENGTH
#define Y_MASK_SIZE PREDICTED_LENGTHS_MAX

#define LEXICON_ZH_FILE "model/lexicon.txt"
#define TOKENS_ZH_FILE "model/tokens.txt"

#define LEXICON_EN_FILE "model/lexicon_en.txt"
#define TOKENS_EN_FILE "model/tokens_en.txt"

#define SDP_RATIO 0.2
#define NOISE_SCALE 0.6 
#define NOISE_SCALE_W 0.8

void middle_process(std::vector<float> log_duration, std::vector<float> input_padding_mask, 
    std::vector<float> &attn, std::vector<float> &output_padding_mask, float speed, int &predicted_lengths_max_real);

#endif
