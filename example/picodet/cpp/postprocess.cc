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

// Reference from FastDeploy(https://github.com/PaddlePaddle/FastDeploy)

#include "picodet.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <set>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <functional>
#include <iostream>

#define LABEL_NALE_TXT_PATH "./model/coco_80_labels_list.txt"

static char *labels[OBJ_CLASS_NUM];

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

static char *readLine(FILE *fp, char *buffer, int *len)
{
    int ch;
    int i = 0;
    size_t buff_len = 0;

    buffer = (char *)malloc(buff_len + 1);
    if (!buffer)
        return NULL; // Out of memory

    while ((ch = fgetc(fp)) != '\n' && ch != EOF)
    {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL)
        {
            free(buffer);
            return NULL; // Out of memory
        }
        buffer = (char *)tmp;

        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';

    *len = buff_len;

    // Detect end
    if (ch == EOF && (i == 0 || ferror(fp)))
    {
        free(buffer);
        return NULL;
    }
    return buffer;
}

static int readLines(const char *fileName, char *lines[], int max_line)
{
    FILE *file = fopen(fileName, "r");
    char *s;
    int i = 0;
    int n = 0;

    if (file == NULL)
    {
        printf("Open %s fail!\n", fileName);
        return -1;
    }

    while ((s = readLine(file, s, &n)) != NULL)
    {
        lines[i++] = s;
        if (i >= max_line)
            break;
    }
    fclose(file);
    return i;
}

static int loadLabelName(const char *locationFilename, char *label[])
{
    printf("load lable %s\n", locationFilename);
    readLines(locationFilename, label, OBJ_CLASS_NUM);
    return 0;
}

int init_post_process()
{
    int ret = 0;
    ret = loadLabelName(LABEL_NALE_TXT_PATH, labels);
    if (ret < 0)
    {
        printf("Load %s failed!\n", LABEL_NALE_TXT_PATH);
        return -1;
    }
    return 0;
}

char *coco_cls_to_name(int cls_id)
{

    if (cls_id >= OBJ_CLASS_NUM)
    {
        return "null";
    }

    if (labels[cls_id])
    {
        return labels[cls_id];
    }

    return "null";
}

void deinit_post_process()
{
    for (int i = 0; i < OBJ_CLASS_NUM; i++)
    {
        if (labels[i] != nullptr)
        {
            free(labels[i]);
            labels[i] = nullptr;
        }
    }
}

float BBoxArea(const float* box, const bool& normalized) {
  if (box[2] < box[0] || box[3] < box[1]) {
    // If coordinate values are is invalid
    // (e.g. xmax < xmin or ymax < ymin), return 0.
    return 0.f;
  } else {
    const float w = box[2] - box[0];
    const float h = box[3] - box[1];
    if (normalized) {
      return w * h;
    } else {
      // If coordinate values are not within range [0, 1].
      return (w + 1) * (h + 1);
    }
  }
}

float JaccardOverlap(const float* box1, const float* box2,
                     const bool& normalized) {
  if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
      box2[3] < box1[1]) {
    return 0.f;
  } else {
    const float inter_xmin = std::max(box1[0], box2[0]);
    const float inter_ymin = std::max(box1[1], box2[1]);
    const float inter_xmax = std::min(box1[2], box2[2]);
    const float inter_ymax = std::min(box1[3], box2[3]);
    float norm = normalized ? 0.0f : 1.0f;
    float inter_w = inter_xmax - inter_xmin + norm;
    float inter_h = inter_ymax - inter_ymin + norm;
    const float inter_area = inter_w * inter_h;
    const float bbox1_area = BBoxArea(box1, normalized);
    const float bbox2_area = BBoxArea(box2, normalized);
    return inter_area / (bbox1_area + bbox2_area - inter_area);
  }
}

template<class T>
bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

void GetMaxScoreIndex(const float* scores, const int& score_size,
                      const float& threshold, const int& top_k,
                      std::vector<std::pair<float, int>>* sorted_indices) {
  for (size_t i = 0; i < score_size; ++i) {
    if (scores[i] > threshold) {
      sorted_indices->push_back(std::make_pair(scores[i], i));
    }
  }
  // Sort the score pair according to the scores in descending order
  std::stable_sort(sorted_indices->begin(), sorted_indices->end(),
                   SortScorePairDescend<int>);
  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < static_cast<int>(sorted_indices->size())) {
    sorted_indices->resize(top_k);
  }
}

void FastNMS(const float* boxes, const float* scores,
                            const int& num_boxes,
                            std::vector<int>* keep_indices) {
  std::vector<std::pair<float, int>> sorted_indices;
  GetMaxScoreIndex(scores, num_boxes, BOX_THRESH, NMS_TOP_K,
                   &sorted_indices);

  float adaptive_threshold = NMS_THRESH;
  while (sorted_indices.size() != 0) {
    const int idx = sorted_indices.front().second;
    bool keep = true;
    for (size_t k = 0; k < keep_indices->size(); ++k) {
      if (!keep) {
        break;
      }
      const int kept_idx = (*keep_indices)[k];
      float overlap =
          JaccardOverlap(boxes + idx * 4, boxes + kept_idx * 4, NORMALIZED);
      keep = overlap <= adaptive_threshold;
    }
    if (keep) {
      keep_indices->push_back(idx);
    }
    sorted_indices.erase(sorted_indices.begin());
  }
}

int NMSForEachSample(
    const float* boxes, const float* scores, int num_boxes, int num_classes,
    std::map<int, std::vector<int>>* keep_indices) 
{
  for (int i = 0; i < num_classes; ++i) {
    const float* score_for_class_i = scores + i * num_boxes;
    FastNMS(boxes, score_for_class_i, num_boxes, &((*keep_indices)[i]));
  }
  int num_det = 0;
  for (auto iter = keep_indices->begin(); iter != keep_indices->end(); ++iter) {
    num_det += iter->second.size();
  }

  if (num_det > OBJ_NUMB_MAX_SIZE) {
    std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
    for (const auto& it : *keep_indices) {
      int label = it.first;
      const float* current_score = scores + label * num_boxes;
      auto& label_indices = it.second;
      for (size_t j = 0; j < label_indices.size(); ++j) {
        int idx = label_indices[j];
        score_index_pairs.push_back(
            std::make_pair(current_score[idx], std::make_pair(label, idx)));
      }
    }
    std::stable_sort(score_index_pairs.begin(), score_index_pairs.end(),
                     SortScorePairDescend<std::pair<int, int>>);
    score_index_pairs.resize(OBJ_NUMB_MAX_SIZE);

    std::map<int, std::vector<int>> new_indices;
    for (size_t j = 0; j < score_index_pairs.size(); ++j) {
      int label = score_index_pairs[j].second.first;
      int idx = score_index_pairs[j].second.second;
      new_indices[label].push_back(idx);
    }
    new_indices.swap(*keep_indices);
    num_det = OBJ_NUMB_MAX_SIZE;
  }
  return num_det;
}

int picodet_post_process(rknn_app_context_t *app_ctx, const float* boxes_data, const float* scores_data, object_detect_result* results) 
{
    uint32_t batch_size =  app_ctx->output_attrs[0].dims[0];

    uint32_t boxes_dim1 = app_ctx->output_attrs[0].dims[1];
    uint32_t boxes_dim2 = app_ctx->output_attrs[0].dims[2];

    uint32_t scores_dim1 = app_ctx->output_attrs[1].dims[1];
    uint32_t scores_dim2 = app_ctx->output_attrs[1].dims[2];

    if(batch_size != 1 || scores_dim1 != OBJ_CLASS_NUM || boxes_dim1 != scores_dim2) {
      printf("Model outputs not support!\n");
      return -1;
    }

    // nms
    int num_nmsed_out = 0;
    std::map<int, std::vector<int>> indices;  // indices kept for each class
    int num = NMSForEachSample(boxes_data, scores_data, boxes_dim1, scores_dim1, &indices);
    num_nmsed_out += num;

    if (num_nmsed_out == 0) {
        return -1;
    }

    // Get boxes
    results->boxes.resize(num_nmsed_out);
    results->scores.resize(num_nmsed_out);
    results->label_ids.resize(num_nmsed_out);

    for (const auto& it : indices) {
      int label = it.first;
      const auto& indices = it.second;
      const float* current_scores_class_ptr = scores_data + label * scores_dim2;
      for (size_t j = 0; j < indices.size(); ++j) {
          results->label_ids.push_back(static_cast<int32_t>(label));
          results->scores.push_back(current_scores_class_ptr[indices[j]]);
          results->boxes.emplace_back(std::array<float, 4>(
              {boxes_data[indices[j] * 4], boxes_data[indices[j] * 4 + 1], 
              boxes_data[indices[j] * 4 + 2], boxes_data[indices[j] * 4 + 3]}));
      }
    }
    return 0;
}
