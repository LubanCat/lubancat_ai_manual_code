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

#include "yolov10.h"
#include "image_utils.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

/*-------------------------------------------
                  Main Function
-------------------------------------------*/

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("%s <model_path> <image_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];

    int ret;
    cv::Mat orig_img, image;
    struct timeval start_time, stop_time;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

#ifndef ENABLE_ZERO_COPY
    ret = init_yolov10_model(model_path, &rknn_app_ctx);
#else
    ret = init_yolov10_zero_copy_model(model_path, &rknn_app_ctx);
#endif
    if (ret != 0)
    {
        printf("init_yolov10_model fail! ret=%d model_path=%s\n", ret, model_path);
        goto out;
    }

    // 读取图片
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    gettimeofday(&start_time, NULL);
    orig_img = cv::imread(image_path);
    if (!orig_img.data) {
        printf("cv::imread %s fail!\n", image_path);
        return -1;
    }
    gettimeofday(&stop_time, NULL);
    printf("OpenCV read an image using %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    cv::cvtColor(orig_img, image, cv::COLOR_BGR2RGB);
    src_image.width  = image.cols;
    src_image.height = image.rows;
    src_image.format = IMAGE_FORMAT_RGB888;
    src_image.virt_addr = (unsigned char*)image.data;

    // rknn推理和处理
    gettimeofday(&start_time, NULL);
    object_detect_result_list od_results;
#ifndef ENABLE_ZERO_COPY
    ret = inference_yolov10_model(&rknn_app_ctx, &src_image, &od_results);
#else
    ret = inference_yolov10_zero_copy_model(&rknn_app_ctx, &src_image, &od_results);
#endif
    if (ret != 0)
    {
        printf("init_yolov10_model fail! ret=%d\n", ret);
        goto out;
    }
    gettimeofday(&stop_time, NULL);
    printf("inference_yolov10_model use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    // 画框和概率
    gettimeofday(&start_time, NULL);
    char text[256];
    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result *det_result = &(od_results.results[i]);
        printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
               det_result->box.left, det_result->box.top,
               det_result->box.right, det_result->box.bottom,
               det_result->prop);
        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        cv::rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 2);
        cv::putText(orig_img, text, cv::Point(x1, y1 - 6), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    }

    gettimeofday(&stop_time, NULL);
    printf("putText and printf use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    // 保存结果
    cv::imwrite("./out.jpg", orig_img);

    // 显示结果
    // cv::imshow("image", image);
    // cv::waitKey(0);

out:
    deinit_post_process();
#ifndef ENABLE_ZERO_COPY
    ret = release_yolov10_model(&rknn_app_ctx);
#else
    ret = release_yolov10_zero_copy_model(&rknn_app_ctx);
#endif
    if (ret != 0)
    {
        printf("release yolov10 model fail! ret=%d\n", ret);
    }
    return 0;
}
