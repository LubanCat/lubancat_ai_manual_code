
#ifndef PARSE_ARGS_H_
#define PARSE_ARGS_H_

#include <iostream>
#include <unordered_set>
#include <sstream>

struct Args {
    std::string input_text = "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。";
    // std::string input_text =  "Mister quilter is the apostle of the middle classes and we are glad to welcome his gospel.";

    std::string output_filename = "audio.wav";

    float speed = 1.0;
    bool disable_bert = true;
    bool add_blank = true;

    std::string language  = "ZH"; // ZH_MIX_EN / ZH / EN 
    int speak_id = 1;             // zh: 1   EN: 0 1 2 3 4

    std::string encoder_path     = "model/encoder-ZH_MIX_EN.rknn";
    std::string decoder_path     = "model/decoder-ZH_MIX_EN.rknn";

    std::string middle_path     = "model/middle_ZH_MIX_EN.onnx";
    std::string bert_path     = "model/bert.rknn";
};

inline void usage(const std::string& prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\n"
              << "options:\n"
              << "  --input_text            Specifies the input text to be processed.\n"
              << "  --encoder_model_path    Encoder model path \n"
              << "  --decoder_model_path    Decoder model path \n"
              << "  --output_filename       Specifies the output audio filename to be generated in the format, For example: audio.wav\n"
              << "  --speed                 Specifies the speed of output audio (default: 1.0).\n"
              << "  --speak_id              (default: 1.0).                                         \n"
              << "  --disable_bert          Indicates whether to disable the BERT model inference (default: ture).\n"
              << "  --language              Specifies the language (ZH_MIX_EN / ZH / EN) for TTS (default: ZH).\n";
}

static bool to_bool(const std::string& s) {
    bool res;
    std::istringstream(s) >> std::boolalpha >> res;
    return res;
}

inline Args parse_args(const std::vector<std::string>& argv) {
    Args args;

    for (size_t i = 1; i < argv.size(); i++) {
        const std::string& arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else if (arg == "--encoder_model_path") {
            args.encoder_path = argv[++i];
        } else if (arg == "--decoder_model_path") {
            args.decoder_path = argv[++i];
        } else if (arg == "--bert_model_path") {
            args.bert_path = argv[++i];
        } else if (arg == "--input_text") {
            args.input_text = argv[++i];
        } else if (arg == "--output_filename") {
            args.output_filename = argv[++i];
        } else if (arg == "--speed") {
            args.speed = std::stof(argv[++i]);
        } else if (arg == "--speak_id") {
            args.speak_id = std::stoi(argv[++i]);
        } else if (arg == "--disable_bert") {
            args.disable_bert = to_bool(argv[++i]);
        } else if (arg == "--language") {
            args.language = argv[++i];
        } else {
            usage(argv[0]);
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return args;
}

inline Args parse_args(int argc, char** argv) {
    std::vector<std::string> argv_vec;
    argv_vec.reserve(argc);

    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(argv[i]);
    }

    return parse_args(argv_vec);
}

#endif
