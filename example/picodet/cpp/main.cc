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

#include <algorithm>
#include <array>
#include <vector>

#include "picodet.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "easy_timer.h"

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
    TIMER timer;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    float score_threshold = BOX_THRESH;
    object_detect_result od_results;

    ret = init_post_process();
    if (ret != 0)
    {
        printf("init_post_process fail!\n");
        goto out;
    }

    ret = init_picodet_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_picodet_model fail! ret=%d model_path=%s\n", ret, model_path);
        goto out;
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    ret = read_image(image_path, &src_image);
    if (ret != 0)
    {
        printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
        goto out;
    }

    timer.tik();
    ret = inference_picodet_model(&rknn_app_ctx, &src_image, &od_results);
    if (ret != 0)
    {
        printf("inference_picodet_model fail! ret=%d\n", ret);
        goto out;
    }
    timer.tok();
    timer.print_time("inference_picodet_model");

    // 画框和概率
    char text[256];

    if (od_results.boxes.empty()) {
        printf("no object detected\n");
        goto out;
    }

    for (size_t i = 0; i < od_results.boxes.size(); ++i) {

        if (od_results.scores[i] < score_threshold) {
            continue;
        }

        int x1 = static_cast<int>(od_results.boxes[i][0]);
        int y1 = static_cast<int>(od_results.boxes[i][1]);
        int x2 = static_cast<int>(od_results.boxes[i][2]);
        int y2 = static_cast<int>(od_results.boxes[i][3]);
        int box_h = y2 - y1;
        int box_w = x2 - x1;
        
        printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(od_results.label_ids[i]), x1, y1, x2, y2,
        od_results.scores[i]);

        draw_rectangle(&src_image, x1, y1, box_w, box_h, COLOR_BLUE, 3);

        sprintf(text, "%s %.1f%%", coco_cls_to_name(od_results.label_ids[i]), od_results.scores[i] * 100);
        draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
    }

    write_image("out.png", &src_image);

out:
    deinit_post_process();

    ret = release_picodet_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_picodet_model fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }

    return 0;
}