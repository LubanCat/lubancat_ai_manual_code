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

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <algorithm>

#include "yolo26_depth.h"
#include "image_utils.h"
#include "file_utils.h"
#include "easy_timer.h"

static inline int n_round(float x) { return static_cast<int>(std::nearbyint(x)); }

cv::Mat letterbox(const cv::Mat& src, int dst_h, int dst_w, letterbox_t& lb, bool use_letterbox = true) {
    int new_w, new_h, dw, dh;

    if (use_letterbox) {
        // 等比缩放/放大
        float r = std::min((float)dst_h / src.rows, (float)dst_w / src.cols);
        new_w = n_round(src.cols * r);
        new_h = n_round(src.rows * r);
        dw = dst_w - new_w;
        dh = dst_h - new_h;
        lb.scale = r;
    } else {
        // 直接拉伸到目标尺寸
        new_w = dst_w;
        new_h = dst_h;
        dw = 0;
        dh = 0;
        lb.scale = 1.0f;
    }

    // 居中分配 (dw=dh=0 时 top/left/bottom/right 全为 0)
    int top    = n_round(dh / 2.0f - 0.1f);
    int left   = n_round(dw / 2.0f - 0.1f);
    int bottom = dh - top;
    int right  = dw - left;

    // resize (双线性)
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    lb.x_pad = left;   // 拉伸模式下 = 0
    lb.y_pad = top;    // 拉伸模式下 = 0

    if (top == 0 && bottom == 0 && left == 0 && right == 0)
        return resized;

    // 填充114 灰边
    cv::Mat out;
    cv::copyMakeBorder(resized, out, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return out;
}


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("%s <model_path> <image_path> \n", argv[0]);
        printf("Example: ./yolo26_depth_demo <model_path> <image_path>\n");
        return -1;
    }

    int ret;
    TIMER timer;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));   

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    
    letterbox_t lb;
    memset(&lb, 0, sizeof(letterbox_t));

    const char *model_path = argv[1];
    const char *image_path = argv[2];

    cv::Mat depth_maps;
    cv::Mat depth_heat, depth_overlay;
    cv::Mat bgr_img, rgb_img, input_img;

    std::vector<float> output_buffer;

    timer.tik();
    ret = init_yolo26_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolo26_model fail! ret=%d model_path=%s\n", ret, model_path);
        goto out;
    }
    timer.tok();
    timer.print_time("init_yolo26_model");

    // 读取图片和预处理
    timer.tik();
    bgr_img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr_img.empty())
    {
        printf("Failed to read image: %s\n", image_path);
        goto out;
    }
    cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
    printf("Read image: %s, size: %dx%d\n", image_path, bgr_img.cols, bgr_img.rows);
    
    // pre-processing
    input_img = letterbox(rgb_img, rknn_app_ctx.model_height, rknn_app_ctx.model_width, lb, false);
    timer.tok();
    timer.print_time("read_image and preprocess");

    rknn_app_ctx.input_image_width = bgr_img.cols;
    rknn_app_ctx.input_image_height = bgr_img.rows;

    // 推理
    ret = inference_yolo26_model(&rknn_app_ctx, input_img, output_buffer);
    if (ret != 0)
    {
        printf("inference_yolo26_model fail! ret=%d\n", ret);
        goto out;
    }

    // post process
    timer.tik();
    ret =  post_process_pre(&rknn_app_ctx, output_buffer, &lb, depth_maps);
    if (ret < 0)
    {
        printf("post_process_pre fail! ret=%d\n", ret);
        goto out;
    }
    timer.tok();
    timer.print_time("post process");

    // 保存结果
    timer.tik();
    depth_heat = colorize_depth(depth_maps);
    cv::imwrite("depth_heat.png", depth_heat);
    depth_overlay = render_depth_overlay(bgr_img, depth_heat);
    cv::imwrite("depth_overlay.png", depth_overlay);
    timer.tok();
    timer.print_time("save results");

out:
    ret = release_yolo26_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolo26_model fail! ret=%d\n", ret);
    }

    return 0;
}

