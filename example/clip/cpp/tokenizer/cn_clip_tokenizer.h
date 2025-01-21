#ifndef _CN_CLIP_TOKENIZER_H
#define _CN_CLIP_TOKENIZER_H

#include <string>
#include <regex>
#include <set>
#include <codecvt>
#include <locale>
#include <map>
#include <vector>

std::vector<std::string> whitespace_tokenize(std::string text);

std::map<std::string, int> read_vocab(const char *filename);

class BasicTokenizer
{
public:
    bool do_lower_case_;
    std::vector<std::string> never_split_;

    BasicTokenizer(bool do_lower_case = false,
                   std::vector<std::string> never_split = {"[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"})
    {
        do_lower_case_ = do_lower_case;
        never_split_ = never_split;
    }

    std::string _clean_text(std::string text);

    std::vector<std::string> _run_split_on_punc(std::string text);

    std::string _run_strip_accents(std::string text);

    std::string _tokenize_chinese_chars(std::string text);

    std::string utf8chr(int cp);

    bool _is_chinese_char(int cp);

    std::vector<std::string> tokenize(std::string text);

    void truncate_sequences(
            std::vector<std::string> &textA, std::vector<std::string> &textB, const char *truncation_strategy, int max_seq_length);
};

class WordpieceTokenizer
{
public:
    std::map<std::string, int> vocab_;
    std::string unk_token_;
    int max_input_chars_per_word_;

    WordpieceTokenizer() {};

    WordpieceTokenizer(std::map<std::string, int> vocab, std::string unk_token = "[UNK]", int max_input_chars_per_word = 100)
    {
        vocab_ = vocab;
        unk_token_ = unk_token;
        max_input_chars_per_word_ = max_input_chars_per_word;
    }

    void init(std::map<std::string, int> vocab);

    std::vector<std::string> tokenize(std::string text);
};

class ChineseCLIPTokenizer
{
public:
    std::map<std::string, int> vocab;
    int max_seq_length;
    BasicTokenizer basic_tokenizer;
    WordpieceTokenizer wordpiece_tokenizer;

    ChineseCLIPTokenizer(const char *vocab_file, int max_len = 512)
    {
        vocab = read_vocab(vocab_file);
        wordpiece_tokenizer.init(vocab);
        max_seq_length = max_len;
    }

    std::vector<int> tokenize(std::string text);

    std::vector<int> convert_tokens_to_ids(std::vector<std::string> tokens);
};


#endif // _CN_CLIP_TOKENIZER_H