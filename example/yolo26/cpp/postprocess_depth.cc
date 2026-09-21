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

#include "yolo26_depth.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 深度图伪彩化
cv::Mat colorize_depth(const cv::Mat& depth, bool disparity,          // 视差模式 (近=暖)
                       double p_lo, double p_hi,                      // 分位数百分比
                       int cmap )
{
    CV_Assert(depth.type() == CV_32FC1);  

    // 收集有效值
    cv::Mat v = cv::Mat::zeros(depth.size(), CV_32FC1);
    std::vector<float> pool;
    pool.reserve(depth.total());
    for (int y = 0; y < depth.rows; ++y) {
        const float* d = depth.ptr<float>(y);
        float* vv = v.ptr<float>(y);
        for (int x = 0; x < depth.cols; ++x)
            if (d[x] > 0) { vv[x] = disparity ? 1.0f / d[x] : d[x]; pool.push_back(vv[x]); }
    }

    // 计算分位数 vmin/vmax (排除噪声)
    float vmin = 0.f, vmax = 1.f;
    if (!pool.empty()) {
        std::sort(pool.begin(), pool.end());
        auto pct = [&](double p) {
            double t = p / 100.0 * (pool.size() - 1);
            int i = (int)t;
            double f = t - i;
            return (float)(pool[i] + f * (pool[std::min(i + 1, (int)pool.size() - 1)] - pool[i]));
        };
        vmin = pct(p_lo);
        vmax = pct(p_hi);
    }
    if (vmax <= vmin) vmax = vmin + 1e-6f;

    // 归一化 → 256 级灰度索引
    cv::Mat idx(depth.size(), CV_8UC1);
    const float inv = 255.0f / (vmax - vmin);
    for (int y = 0; y < depth.rows; ++y) {
        const float* vv = v.ptr<float>(y);
        uchar* o = idx.ptr<uchar>(y);
        for (int x = 0; x < depth.cols; ++x) {
            float dn = (vv[x] - vmin) * inv;              // 负值(远)截 0, 超界(近)截 255
            o[x] = (uchar)std::min(255.f, std::max(0.f, dn));
        }
    }

    // 伪彩 + 无效像素置黑
    cv::Mat heat, invalid;
    cv::applyColorMap(idx, heat, cmap);                   // 输出 BGR uint8
    cv::compare(depth, 0, invalid, cv::CMP_LE);
    heat.setTo(cv::Scalar(0, 0, 0), invalid);
    return heat;
}

// 热力图叠加原图
cv::Mat render_depth_overlay(const cv::Mat& bgr_orig,     // 原图 BGR uint8
                             const cv::Mat& heat,
                             double alpha) {
    if (heat.size() != bgr_orig.size())
        cv::resize(heat, heat, bgr_orig.size());
    cv::Mat out;
    cv::addWeighted(bgr_orig, 1 - alpha, heat, alpha, 0, out);
    return out;
}


int post_process_pre(rknn_app_context_t *app_ctx, std::vector<float>& output, letterbox_t *letter_box, cv::Mat& depth_maps)
{
    cv::Mat depth_heat, depth_overlay;

    // 1*1*768*768
    int output_H = app_ctx->output_attrs[0].dims[2];
    int output_W = app_ctx->output_attrs[0].dims[3];

    if(app_ctx->input_image_width == output_W && app_ctx->input_image_height == output_H)
    {
        // 原始图像和模型输入尺寸一致，直接输出
        depth_maps = cv::Mat(output_H, output_W, CV_32FC1, output.data());
    }
    else
    {
        // Mat (768x768, float32 单通道)
        cv::Mat m(output_H, output_W, CV_32FC1, output.data());

        if(letter_box->x_pad == 0 && letter_box->y_pad == 0)
        {   
            // 预处理未使用letterbox，直接resize
            cv::resize(m, depth_maps, cv::Size(app_ctx->input_image_width, app_ctx->input_image_height), 0, 0, cv::INTER_LINEAR);
        }
        else
        {
            // 预处理使用letterbox，推理输出需要裁剪掉pad部分再resize
            cv::Rect roi(letter_box->x_pad, letter_box->y_pad, output_W - 2 * letter_box->x_pad, output_H - 2 * letter_box->y_pad);
            cv::Mat cropped = m(roi).clone();
            cv::resize(cropped, depth_maps, cv::Size(app_ctx->input_image_width, app_ctx->input_image_height), 0, 0, cv::INTER_LINEAR);
        }
    }

    // 获取最大最小值及其位置
    // double minVal, maxVal;
    // cv::Point minLoc, maxLoc;
    // cv::minMaxLoc(depth_maps, &minVal, &maxVal, &minLoc, &maxLoc);

    // 打印最大最小值及其位置
    // printf("Min value: %f at (%d, %d)\n", minVal, minLoc.x, minLoc.y);
    // printf("Max value: %f at (%d, %d)\n", maxVal, maxLoc.x, maxLoc.y);

    return 0;
}