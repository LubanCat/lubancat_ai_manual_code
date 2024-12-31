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

#include "yolo11.h"
#include "image_utils.h"
#include "file_utils.h"

#if defined(RV1106_1103) 
    #include "dma_alloc.hpp"
#endif

#include <opencv2/opencv.hpp>

static const unsigned char colors[19][3] = {
    {54, 67, 244},
    {99, 30, 233},
    {176, 39, 156},
    {183, 58, 103},
    {181, 81, 63},
    {243, 150, 33},
    {244, 169, 3},
    {212, 188, 0},
    {136, 150, 0},
    {80, 175, 76},
    {74, 195, 139},
    {57, 220, 205},
    {59, 235, 255},
    {7, 193, 255},
    {0, 152, 255},
    {34, 87, 255},
    {72, 85, 121},
    {158, 158, 158},
    {139, 125, 96}
};

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("%s <model path> <camera device id/video path>\n", argv[0]);
        printf("Usage: %s  yolov11s.rknn  0 \n", argv[0]);
        printf("Usage: %s  yolov11s.rknn /path/xxxx.mp4\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *device_path = argv[2];

    int ret;
    cv::Mat frame, image;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));

    cv::VideoCapture cap;
    // 摄像头
    if (isdigit(device_path[0])) {
        int camera_id = atoi(argv[2]);
        cap.open(camera_id);

        if (!cap.isOpened()) {
            printf("Error: Could not open camera.\n");
            return -1;
        }
        // cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        // cap.set(cv::CAP_PROP_CONVERT_RGB,1);
        // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);    // 设置宽度
        // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);  // 设置长度
    } else {
        // 视频文件或者其他
        cap.open(argv[2]);
        if (!cap.isOpened()) {  
            printf("Error: Could not open video file.\n");
            return -1;
        }
    }

    init_post_process();
    ret = init_yolo11_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolo11_model fail! ret=%d model_path=%s\n", ret, model_path);
        goto out;
    }

    while(true) {
        if (!cap.read(frame)) {  
            printf("cap read frame fail!\n");
            break;
        }

        cv::cvtColor(frame, image, cv::COLOR_BGR2RGB);
        src_image.width  = image.cols;
        src_image.height = image.rows;
        src_image.format = IMAGE_FORMAT_RGB888;
        src_image.virt_addr = (unsigned char*)image.data;

        // rknn推理和处理
        object_detect_result_list od_results;
        ret = inference_yolo11_model(&rknn_app_ctx, &src_image, &od_results);
        if (ret != 0)
        {
            printf("init_yolov10_model fail! ret=%d\n", ret);
            goto out;
        }

        // 画框和概率
        int color_index = 0;
        char text[256];
        for (int i = 0; i < od_results.count; i++)
        {
            const unsigned char* color = colors[color_index % 19];
            cv::Scalar cc(color[0], color[1], color[2]);
            color_index++;

            object_detect_result *det_result = &(od_results.results[i]);
            printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
                det_result->box.left, det_result->box.top,
                det_result->box.right, det_result->box.bottom,
                det_result->prop);
            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);

            cv::rectangle(frame, cv::Rect(cv::Point(det_result->box.left, det_result->box.top), 
                            cv::Point(det_result->box.right, det_result->box.bottom)), cc, 2);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = det_result->box.left;
            int y = det_result->box.top - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > frame.cols)
                x = frame.cols - label_size.width;

            cv::rectangle(frame, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),cc,-1);
            cv::putText(frame, text, cv::Point(x, y + label_size.height),cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
        }

        // 显示结果
        cv::imshow("yolo11", frame);

        char c = cv::waitKey(1);
        if (c == 27) { // ESC
            break;
        }
    }

out:
    deinit_post_process();

    ret = release_yolo11_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolo11_model fail! ret=%d\n", ret);
    }
    return 0;
}
