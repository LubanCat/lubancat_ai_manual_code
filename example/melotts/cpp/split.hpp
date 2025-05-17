// from https://github.com/ml-inory/melotts.axera/blob/main/cpp/src/split_utils.hpp

#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <cctype>
#include <iterator>

using namespace std;

// 判断是否是UTF-8字符的后续字节
inline bool is_utf8_continuation_byte(unsigned char c) {
    return (c & 0xC0) == 0x80;
}

// 计算UTF-8字符串的字符数（非字节数）
size_t utf8_strlen(const string& str) {
    size_t len = 0;
    for (size_t i = 0; i < str.size(); ) {
        unsigned char c = str[i];
        if ((c & 0x80) == 0) { // ASCII字符
            i += 1;
        } else if ((c & 0xE0) == 0xC0) { // 2字节UTF-8
            i += 2;
        } else if ((c & 0xF0) == 0xE0) { // 3字节UTF-8（包括大部分中文）
            i += 3;
        } else if ((c & 0xF8) == 0xF0) { // 4字节UTF-8
            i += 4;
        } else {
            i++; // 无效UTF-8，跳过
        }
        len++;
    }
    return len;
}

// 合并短句的英文版本
vector<string> merge_short_sentences_en(const vector<string>& sens) {
    vector<string> sens_out;
    for (const auto& s : sens) {
        // 如果前一个句子太短（<=2个单词），就与当前句子合并
        if (!sens_out.empty()) {
            istringstream iss(sens_out.back());
            int word_count = distance(istream_iterator<string>(iss), istream_iterator<string>());
            if (word_count <= 2) {
                sens_out.back() += " " + s;
                continue;
            }
        }
        sens_out.push_back(s);
    }
    
    // 处理最后一个句子如果太短的情况
    if (!sens_out.empty() && sens_out.size() > 1) {
        istringstream iss(sens_out.back());
        int word_count = distance(istream_iterator<string>(iss), istream_iterator<string>());
        if (word_count <= 2) {
            sens_out[sens_out.size()-2] += " " + sens_out.back();
            sens_out.pop_back();
        }
    }
    
    return sens_out;
}

// 合并短句的中文版本
vector<string> merge_short_sentences_zh(const vector<string>& sens) {
    vector<string> sens_out;
    for (const auto& s : sens) {
        // 如果前一个句子太短（<=2个字符），就与当前句子合并
        if (!sens_out.empty() && utf8_strlen(sens_out.back()) <= 2) {
            sens_out.back() += " " + s;
        } else {
            sens_out.push_back(s);
        }
    }
    
    // 处理最后一个句子如果太短的情况
    if (!sens_out.empty() && sens_out.size() > 1 && utf8_strlen(sens_out.back()) <= 2) {
        sens_out[sens_out.size()-2] += " " + sens_out.back();
        sens_out.pop_back();
    }
    
    return sens_out;
}

// 替换字符串中的子串
string replace_all(const string& input, const string& from, const string& to) {
    string result = input;
    size_t pos = 0;
    while ((pos = result.find(from, pos)) != string::npos) {
        result.replace(pos, from.length(), to);
        pos += to.length();
    }
    return result;
}

// 分割拉丁语系文本（英文、法文、西班牙文等）
vector<string> split_sentences_latin(const string& text, int min_len = 10) {
    string processed = text;
    
    // 替换中文标点为英文标点
    processed = replace_all(processed, "。", ".");
    processed = replace_all(processed, "！", ".");
    processed = replace_all(processed, "？", ".");
    processed = replace_all(processed, "；", ".");
    processed = replace_all(processed, "，", ",");
    processed = replace_all(processed, "“", "\"");
    processed = replace_all(processed, "”", "\"");
    processed = replace_all(processed, "‘", "'");
    processed = replace_all(processed, "’", "'");
    
    // 移除特定字符
    string chars_to_remove = "<>()[]\"«»";
    for (char c : chars_to_remove) {
        processed.erase(remove(processed.begin(), processed.end(), c), processed.end());
    }
    
    // 分割句子（简化版，按句号分割）
    vector<string> sentences;
    size_t start = 0;
    size_t end = processed.find('.');
    
    while (end != string::npos) {
        string sentence = processed.substr(start, end - start);
        // 去除前后空白
        sentence.erase(sentence.begin(), find_if(sentence.begin(), sentence.end(), [](int ch) { return !isspace(ch); }));
        sentence.erase(find_if(sentence.rbegin(), sentence.rend(), [](int ch) { return !isspace(ch); }).base(), sentence.end());
        if (!sentence.empty()) {
            sentences.push_back(sentence);
        }
        start = end + 1;
        end = processed.find('.', start);
    }
    
    // 添加最后一部分
    if (start < processed.size()) {
        string sentence = processed.substr(start);
        sentence.erase(sentence.begin(), find_if(sentence.begin(), sentence.end(), [](int ch) { return !isspace(ch); }));
        sentence.erase(find_if(sentence.rbegin(), sentence.rend(), [](int ch) { return !isspace(ch); }).base(), sentence.end());
        if (!sentence.empty()) {
            sentences.push_back(sentence);
        }
    }
    
    return merge_short_sentences_en(sentences);
}

// 分割中文文本
vector<string> split_sentences_zh(const string& text, int min_len = 10) {
    string processed = text;
    
    // 替换中文标点为英文标点
    processed = replace_all(processed, "。", ".");
    processed = replace_all(processed, "！", ".");
    processed = replace_all(processed, "？", ".");
    processed = replace_all(processed, "；", ".");
    processed = replace_all(processed, "，", ",");
    
    // 将文本中的换行符、空格和制表符替换为空格
    processed = replace_all(processed, "\n", " ");
    processed = replace_all(processed, "\t", " ");
    processed = replace_all(processed, "  ", " "); // 多个空格合并为一个
    
    // 在标点符号后添加一个特殊标记用于分割
    string punctuation = ".,!?;";
    for (char c : punctuation) {
        string from(1, c);
        string to = from + " $#!";
        processed = replace_all(processed, from, to);
    }
    
    // 分割句子
    vector<string> sentences;
    size_t start = 0;
    size_t end = processed.find("$#!");
    
    while (end != string::npos) {
        string sentence = processed.substr(start, end - start);
        // 去除前后空白
        sentence.erase(sentence.begin(), find_if(sentence.begin(), sentence.end(), [](int ch) { return !isspace(ch); }));
        sentence.erase(find_if(sentence.rbegin(), sentence.rend(), [](int ch) { return !isspace(ch); }).base(), sentence.end());
        if (!sentence.empty()) {
            sentences.push_back(sentence);
        }
        start = end + 3; // "$#!" 长度为3
        end = processed.find("$#!", start);
    }
    
    // 添加最后一部分
    if (start < processed.size()) {
        string sentence = processed.substr(start);
        sentence.erase(sentence.begin(), find_if(sentence.begin(), sentence.end(), [](int ch) { return !isspace(ch); }));
        sentence.erase(find_if(sentence.rbegin(), sentence.rend(), [](int ch) { return !isspace(ch); }).base(), sentence.end());
        if (!sentence.empty()) {
            sentences.push_back(sentence);
        }
    }
    
    // 按最小长度合并句子
    vector<string> new_sentences;
    vector<string> new_sent;
    int count_len = 0;
    
    for (size_t i = 0; i < sentences.size(); ++i) {
        new_sent.push_back(sentences[i]);
        count_len += utf8_strlen(sentences[i]);
        if (count_len > min_len || i == sentences.size() - 1) {
            count_len = 0;
            ostringstream oss;
            for (size_t j = 0; j < new_sent.size(); ++j) {
                if (j != 0) oss << " ";
                oss << new_sent[j];
            }
            new_sentences.push_back(oss.str());
            new_sent.clear();
        }
    }
    
    return merge_short_sentences_zh(new_sentences);
}

// 主分割函数
vector<string> split_sentence(const string& text, int min_len = 10, const string& language_str = "EN") {
    if (language_str == "EN" || language_str == "FR" || language_str == "ES" || language_str == "SP") {
        return split_sentences_latin(text, min_len);
    } else {
        return split_sentences_zh(text, min_len);
    }
}