#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "mobileclip_demo.h"
#include "clip_tokenizer.h"

#include "easy_timer.h"

#define SEQUENCE_LENGTH 77

int main(int argc, char* argv[]) {
    if (argc != 5)
    {
        std::cerr << argv[0] <<"<image_model_path> <image_path> <text_model_path> <text_path>" << std::endl;
        return -1;
    }

    TIMER print_out;

    MobileClipModelConfig config;
    config.img_model_path = argv[1];
    config.text_model_path = argv[3];
    config.image_size = 256;
    config.debug = true;

    CLIPTokenizer mobileclip_tokenizer;
    MobileClip mobileclip(config);

    // 读取图片
    std::string image_path = argv[2];
    cv::Mat image = cv::imread(image_path);
    // 检查图片是否成功加载
    if (image.empty()) {
        std::cerr << "Could not open image : " << image_path << std::endl;
        return -1;
    }

    // 图像推理
    print_out.tik();
    cv::Mat image_feature = mobileclip.mobileclip_image_infer(image);
    print_out.tok();
    print_out.print_time("Image inference time");

    // 标签读取
    std::ifstream text_file(argv[4]);
    std::string line;
    std::vector<std::string> rtext;
    if (!text_file.is_open()) {
        std::cerr << "Could not open text!" << std::endl;
        return -1;
    }
    while (std::getline(text_file, line)) {
        rtext.push_back(line);
    }
    text_file.close();

    print_out.tik();
    std::vector<std::vector<int>> tokens;
    for (const auto &t : rtext) {
        tokens.push_back(mobileclip_tokenizer.tokenize(t, SEQUENCE_LENGTH, true));
    }
    
    // 文本推理
    std::vector<cv::Mat> text_features;
    for (const auto &token : tokens) {
        text_features.push_back(mobileclip.mobileclip_text_infer(token));
    }
    print_out.tok();
    print_out.print_time("text_features time");

    // 计算相似度
    print_out.tik();
    std::vector<double> similarities;
    for (const auto &text_feature : text_features) {
        double similarity = image_feature.dot(text_feature);
        similarities.push_back(similarity * 100);
    }
    
    // softmax
    double sum = 0.0;
    for (const auto &sim : similarities) {
        sum += exp(sim);
    }
    for (auto &sim : similarities) {
        sim = exp(sim) / sum;
    }
    print_out.tok();
    print_out.print_time("Similarity calculation time");

    // 输出结果
    std::cout << std::fixed;
    std::cout.precision(6);
    std::cout << "Image: " << image_path << std::endl;
    for (size_t i = 0; i < rtext.size(); ++i) {
        std::cout << "Text: " << rtext[i] << ", Prob: " << similarities[i] << std::endl;
    }
}
