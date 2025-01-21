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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "owlvit.h"
#include "file_utils.h"
#include "image_utils.h"
#include "image_drawing.h"

#include "easy_timer.h"

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc != 5)
    {
        printf("%s <owlvit_text_model_path> <owlvit_image_model_path> <text_path> <image_path> \n", argv[0]);
        return -1;
    }

    const char *text_model_path = argv[1];
    const char *image_model_path = argv[2];
    const char *text_path = argv[3];
    const char *img_path = argv[4];

    int ret;
    TIMER print_out;
    rknn_owlvit_context_t owlvit_ctx;
    memset(&owlvit_ctx, 0, sizeof(rknn_owlvit_context_t));

    printf("--> init owlvit text model\n");
    ret = init_owlvit_model(&owlvit_ctx, text_model_path, image_model_path);
    if (ret != 0)
    {
        printf("init owlvit model fail! ret=%d\n", ret);
        return -1;
    }

    int text_lines;
    char** input_texts = read_lines_from_file(text_path, &text_lines);
    if (input_texts == NULL)
    {
        printf("read input texts fail! ret=%d text_path=%s\n", ret, text_path);
        return -1;
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    ret = read_image(img_path, &src_image);
    if (ret != 0)
    {
        printf("read image fail! ret=%d image_path=%s\n", ret, img_path);
        return -1;
    }

    object_detect_result_list od_results;
    printf("--> inference model\n");
    print_out.tik();
    ret = inference_owlvit_model(&owlvit_ctx, &src_image, input_texts, text_lines, &od_results);
    if (ret != 0)
    {
        printf("inference fail! ret=%d\n", ret);
        goto out;
    }
    print_out.tok();
    print_out.print_time("inference_owlvit_model");

    char text[256];
    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result *det_result = &(od_results.results[i]);
        printf("%s @ (%d %d %d %d) %.3f\n", input_texts[det_result->text_id],
               det_result->box.left, det_result->box.top,
               det_result->box.right, det_result->box.bottom,
               det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

        sprintf(text, "%s %.1f%%", input_texts[det_result->text_id], det_result->prop * 100);
        draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
    }

    write_image("out.png", &src_image);

out:
    ret = release_owlvit_model(&owlvit_ctx);
    if (ret != 0)
    {
        printf("release_owlvit_model fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL)
    {

        free(src_image.virt_addr);
    }

    if (input_texts != NULL)
    {
        free_lines(input_texts, text_lines);
    }
    return 0;
}
