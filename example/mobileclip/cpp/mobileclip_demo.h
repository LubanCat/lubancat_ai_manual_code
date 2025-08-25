#ifndef MOBILECLIP_DEMO_H
#define MOBILECLIP_DEMO_H

#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include <opencv2/opencv.hpp>

#include "rknn_api.h"

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;

    int model_channel;
    int model_width;  // text_batch_size
    int model_height;  // sequence_length
} rknn_clip_context;

struct MobileClipModelConfig {
    std::string rknpu_version;
    std::string img_model_path;
    std::string text_model_path;
    int image_size = 256;
    bool debug = false;
};

class MobileClip {
public:
    explicit MobileClip(const MobileClipModelConfig & config);
    ~MobileClip();

    cv::Mat mobileclip_image_infer(cv::Mat img);
    cv::Mat mobileclip_text_infer(std::vector<int> tokens);

private:
    void init_mobileclip_model(rknn_clip_context* clip_ctx, const std::string &model_path);
    void release_clip_model();

    rknn_clip_context encode_image_ctx;
    rknn_clip_context encode_text_ctx;
    int image_size = 256;
    bool debug = false;
};

#endif //MOBILECLIP_DEMO_H