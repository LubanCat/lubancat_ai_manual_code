// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
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

#include "ppocr_system.h"
#include "image_utils.h"
#include <opencv2/opencv.hpp>

#define INDENT "    "
#define THRESHOLD 0.3                                       // pixel score threshold
#define BOX_THRESHOLD 0.6                            // box score threshold
#define USE_DILATION false                               // whether to do dilation, true or false
#define DB_SCORE_MODE "slow"                        // slow or fast. slow for polygon mask; fast for rectangle mask
#define DB_BOX_TYPE "poly"                                // poly or quad. poly for returning polygon box; quad for returning rectangle box
#define DB_UNCLIP_RATIO 1.5                          // unclip ratio for poly type

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    char* det_model_path = NULL;
    char* cls_model_path = NULL;
    char* rec_model_path = NULL;
    char* image_path = NULL;

    if(argc == 4) {
        det_model_path = argv[1];
        rec_model_path = argv[2];
        image_path = argv[3];
    } else if(argc == 5) {
        det_model_path = argv[1];
        cls_model_path = argv[2];
        rec_model_path = argv[3];
        image_path = argv[4];
    } else {
        printf("%s <det_model_path> <rec_model_path> <image_path>\n", argv[0]);
        printf("or %s <det_model_path> <cls_model_path> <rec_model_path> <image_path>\n", argv[0]);
        return -1;
    }  

    int ret;
    cv::Mat orig_img, image;
    ppocr_system_app_context rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(ppocr_system_app_context));

    ret = init_ppocr_model(det_model_path, &rknn_app_ctx.det_context);
    if (ret != 0) {
        printf("init_ppocr_model fail! ret=%d det_model_path=%s\n", ret, det_model_path);
        return -1;
    }

    if (cls_model_path != NULL){
        ret = init_ppocr_model(cls_model_path, &rknn_app_ctx.cls_context);
        if (ret != 0) {
            printf("init_ppocr_model fail! ret=%d det_model_path=%s\n", ret, det_model_path);
            return -1;
        }
    }

    ret = init_ppocr_model(rec_model_path, &rknn_app_ctx.rec_context);
    if (ret != 0) {
        printf("init_ppocr_model fail! ret=%d rec_model_path=%s\n", ret, rec_model_path);
        return -1;
    }

    ppocr_text_recog_array_result_t results;
    ppocr_det_postprocess_params params;
    params.threshold = THRESHOLD;
    params.box_threshold = BOX_THRESHOLD;
    params.use_dilate = USE_DILATION;
    params.db_score_mode = DB_SCORE_MODE;
    params.db_box_type = DB_BOX_TYPE;
    params.db_unclip_ratio = DB_UNCLIP_RATIO;
    const unsigned char blue[] = {0, 0, 255};

    // OpenCV读取图片
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    orig_img = cv::imread(image_path);
    if (!orig_img.data) {
        printf("cv::imread %s fail!\n", image_path);
        return -1;
    }

    cv::cvtColor(orig_img, image, cv::COLOR_BGR2RGB);
    src_image.width  = image.cols;
    src_image.height = image.rows;
    src_image.format = IMAGE_FORMAT_RGB888;
    src_image.virt_addr = (unsigned char*)image.data;

    ret = inference_ppocr_system_model(&rknn_app_ctx, &src_image, &params, &results);
    if (ret != 0) {
        printf("inference_ppocr_system_model fail! ret=%d\n", ret);
        goto out;
    }

    // Draw Objects
    printf("DRAWING OBJECT\n");
    for (int i = 0; i < results.count; i++)
    {
        printf("[%d] @ [(%d, %d), (%d, %d), (%d, %d), (%d, %d)]\n", i,
            results.text_result[i].box.left_top.x, results.text_result[i].box.left_top.y, results.text_result[i].box.right_top.x, results.text_result[i].box.right_top.y, 
            results.text_result[i].box.right_bottom.x, results.text_result[i].box.right_bottom.y, results.text_result[i].box.left_bottom.x, results.text_result[i].box.left_bottom.y);

        rectangle(orig_img, cv::Point(results.text_result[i].box.left_top.x, results.text_result[i].box.left_top.y), cv::Point(results.text_result[i].box.right_bottom.x, results.text_result[i].box.right_bottom.y), cv::Scalar(255, 0, 0, 255), 1);
        printf("regconize result: %s, score=%f\n", results.text_result[i].text.str, results.text_result[i].text.score);
    }
    cv::imwrite("out.jpg", orig_img);

out:
    ret = release_ppocr_model(&rknn_app_ctx.det_context);
    if (ret != 0) {
        printf("release_ppocr_model det_context fail! ret=%d\n", ret);
    }
    ret = release_ppocr_model(&rknn_app_ctx.rec_context);
    if (ret != 0) {
        printf("release_ppocr_model rec_context fail! ret=%d\n", ret);
    }
    ret = release_ppocr_model(&rknn_app_ctx.cls_context);
    if (ret != 0) {
        printf("release_ppocr_model cls_context fail! ret=%d\n", ret);
    }
    return 0;
}
