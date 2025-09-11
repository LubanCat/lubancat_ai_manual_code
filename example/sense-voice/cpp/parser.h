
#ifndef PARSE_ARGS_H_
#define PARSE_ARGS_H_

#include <iostream>
#include <sstream>
#include <vector>
#include <string>

struct Args {
    std::string model_path = "./model/model.rknn";    // Default model path
    std::string language  = "auto";                   // zh, en, ja, ko, yue, auto
    std::string audio_path = "./model/en.wav";
    std::string tokens = "./model/tokens.txt";
    int use_itn = 0; // 1 to use inverse text normalization, 0 to not use inverse text normalization
};

inline void usage(const std::string& prog) {
    std::cout << "Usage: " << prog << " -m ./model/model.rknn" << " [options]\n"
              << "\n"
              << "options:\n"
              << "  -m or --model_path      Model path, could be .rknn file.(default:./model/model.rknn)\n"
              << "  --tokens                Path to tokens.txt.(default: ./model/tokens.txt) \n"
              << "  --audio_path            The input wave to be recognized  (default: ./model/en.wav).\n"
              << "  --use-itn               1 to use inverse text normalization, 0 to not use inverse text normalization.(default: 0)\n"
              << "  --language              Tthe language of the input wav file. Supported values: zh, en, ja, ko, yue, auto.(default: auto)\n";
}

inline Args parse_args(const std::vector<std::string>& argv) {
    Args args;

    for (size_t i = 1; i < argv.size(); i++) {
        const std::string& arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else if (arg == "-m" || arg == "--model_path") {
            args.model_path = argv[++i];
        } else if (arg == "--tokens") {
            args.tokens = argv[++i];
        } else if (arg == "--language") {
            args.language = argv[++i];
        } else if (arg == "--audio_path") {
            args.audio_path = argv[++i];
        } else if (arg == "--use-itn") {
            args.use_itn = std::stoi(argv[++i]);
        } else {
            std::cout << "Unknown argument: " << arg << std::endl;
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
