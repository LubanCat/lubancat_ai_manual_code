#ifndef _RKNN_SENSEVOICE_DEMO_PROCESS_H_
#define _RKNN_SENSEVOICE_DEMO_PROCESS_H_

#include "rknn_api.h"
#include "easy_timer.h"

#define SAMPLE_RATE 16000
#define CHUNK_LENGTH 7

#define VOCAB_LEN 25055 
#define OUTPUT_LEN 128 

#define LFR_M 7
#define LFR_N 6
#define N_MELS 80
#define FEATURES_LEN  N_MELS * LFR_M  // 560

#define SPECIA_TOKEN_START 24884

#define INPUT_LENGTH 124

typedef struct
{
    int index;
    char *token;
} VocabEntry;

struct CMVNData {
    std::vector<float> means;
    std::vector<float> vars;
};


int load_cmvn(const std::string& cmvn_file, CMVNData& cmvn_data);
void logits_argmax(float *array, int *index);
int read_vocab(const char *fileName, VocabEntry *vocab);
void audio_preprocess(audio_buffer_t *audio, int &len, CMVNData &cmvn_data, std::vector<float> &data);

#endif //_RKNN_SENSEVOICE_DEMO_PROCESS_H_
