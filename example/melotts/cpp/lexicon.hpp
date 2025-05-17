// https://github.com/ml-inory/melotts.axera/blob/main/cpp/src/Lexicon.hpp
#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <assert.h>
#include <sstream>
#include <algorithm>

std::vector<std::string> split (const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss (s);
    std::string item;
    while (getline (ss, item, delim)) {
        result.push_back (item);
    }
    return result;
}

class Lexicon {
private:
    std::unordered_map<std::string, std::pair<std::vector<int>, std::vector<int>>> lexicon;

public:
    Lexicon(const std::string& lexicon_filename, const std::string& tokens_filename) {
        std::unordered_map<std::string, int> tokens;
        std::ifstream ifs(tokens_filename);
        assert(ifs.is_open());

        std::string line;
        while ( std::getline(ifs, line) ) {
            auto splitted_line = split(line, ' ');
            tokens.insert({splitted_line[0], std::stoi(splitted_line[1])});
        }
        ifs.close();

        ifs.open(lexicon_filename);
        assert(ifs.is_open());
        while ( std::getline(ifs, line) ) {
            auto splitted_line = split(line, ' ');
            std::string word_or_phrase = splitted_line[0];
            size_t phone_tone_len = splitted_line.size() - 1;
            size_t half_len = phone_tone_len / 2;
            std::vector<int> phones, tones;
            for (size_t i = 0; i < phone_tone_len; i++) {
                auto phone_or_tone = splitted_line[i + 1];
                if (i < half_len) {
                    phones.push_back(tokens[phone_or_tone]);
                } else {
                    tones.push_back(std::stoi(phone_or_tone));
                }
            }

            lexicon.insert({word_or_phrase, std::make_pair(phones, tones)});
        }

        lexicon["呣"] = lexicon["母"];
        lexicon["嗯"] = lexicon["恩"];

        const std::vector<std::string> punctuation{"!", "?", "…", ",", ".", "'", "-"};
        for (auto p : punctuation) {
            int i = tokens[p];
            int tone = 0;
            lexicon[p] = std::make_pair(std::vector<int>{i}, std::vector<int>{tone});
        }
        lexicon[" "] = std::make_pair(std::vector<int>{tokens["_"]}, std::vector<int>{0});
    }

    std::vector<std::string> splitEachChar(const std::string& text)
    {
        std::vector<std::string> words;
        std::string input(text);
        int len = input.length();
        int i = 0;
        
        while (i < len) {
        int next = 1;
        if ((input[i] & 0x80) == 0x00) {
            // std::cout << "one character: " << input[i] << std::endl;
        } else if ((input[i] & 0xE0) == 0xC0) {
            next = 2;
            // std::cout << "two character: " << input.substr(i, next) << std::endl;
        } else if ((input[i] & 0xF0) == 0xE0) {
            next = 3;
            // std::cout << "three character: " << input.substr(i, next) << std::endl;
        } else if ((input[i] & 0xF8) == 0xF0) {
            next = 4;
            // std::cout << "four character: " << input.substr(i, next) << std::endl;
        }
        words.push_back(input.substr(i, next));
        i += next;
        }
        return words;
    } 

    bool is_english(std::string s) {
        if (s.size() == 1)
            return (s[0] >= 'A' && s[0] <= 'Z') || (s[0] >= 'a' && s[0] <= 'z');
        else
            return false;
    }

    std::vector<std::string> merge_english(const std::vector<std::string>& splitted_text) {
        std::vector<std::string> words;
        int i = 0;
        while (i < splitted_text.size()) {
            std::string s;
            if (is_english(splitted_text[i])) {
                while (i < splitted_text.size()) {
                    if (!is_english(splitted_text[i])) {
                        break;
                    }
                    s += splitted_text[i];
                    i++;
                }
                // to lowercase
                std::transform(s.begin(), s.end(), s.begin(),
                    [](unsigned char c){ return std::tolower(c); });
                words.push_back(s);
                if (i >= splitted_text.size())
                    break;
            }
            else {
                words.push_back(splitted_text[i]);
                i++;
            }
        }
        return words;
    }

    void convert(const std::string& text, std::vector<int>& phones, std::vector<int>& tones) {
        auto splitted_text = splitEachChar(text);
        auto zh_mix_en = merge_english(splitted_text);
        for (auto c : zh_mix_en) {
            std::string s{c};
            if (s == "，") 
                s = ",";
            else if (s == "。")
                s = ".";
            else if (s == "！")
                s = "!";
            else if (s == "？")
                s = "?";

            auto phones_and_tones = lexicon[" "];
            if (lexicon.find(s) != lexicon.end()) {
                phones_and_tones = lexicon[s];
            }
            phones.insert(phones.end(), phones_and_tones.first.begin(), phones_and_tones.first.end());
            tones.insert(tones.end(), phones_and_tones.second.begin(), phones_and_tones.second.end());
        }
    }

    void convert(const std::string& text, std::vector<int>& phones, std::vector<int>& tones, std::vector<int>& word2ph) {
        auto splitted_text = splitEachChar(text);
        auto zh_mix_en = merge_english(splitted_text);
        for (auto c : zh_mix_en) {
            std::string s{c};
            if (s == "，") 
                s = ",";
            else if (s == "。")
                s = ".";
            else if (s == "！")
                s = "!";
            else if (s == "？")
                s = "?";

            auto phones_and_tones = lexicon[" "];
            if (lexicon.find(s) != lexicon.end()) {
                phones_and_tones = lexicon[s];
            }
            phones.insert(phones.end(), phones_and_tones.first.begin(), phones_and_tones.first.end());
            tones.insert(tones.end(), phones_and_tones.second.begin(), phones_and_tones.second.end());
            word2ph.insert(word2ph.end(), (int)phones_and_tones.first.size());
        }
    }

};