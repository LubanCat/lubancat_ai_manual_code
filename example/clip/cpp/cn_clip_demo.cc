/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cn_clip.h"
#include "image_utils.h"
#include "file_utils.h"

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc != 5)
    {
        printf("%s <image_model_path> <image_path> <text_model_path> <text_path>\n", argv[0]);
        return -1;
    }

    const char *img_model_path = argv[1];
    const char *img_path = argv[2];
    const char *text_model_path = argv[3];
    const char *text_path = argv[4];

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    ret = init_cn_clip_model(img_model_path, text_model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_clip_model fail! ret=%d img_model_path=%s text_model_path=%s\n", ret, img_model_path, text_model_path);
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

    int text_lines;
    char** input_texts = read_lines_from_file(text_path, &text_lines);
    if (input_texts == NULL)
    {
        printf("read input texts fail! ret=%d text_path=%s\n", ret, text_path);
        return -1;
    }

    clip_res out_res;
    ret = inference_cn_clip_model(&rknn_app_ctx, &src_image, input_texts, text_lines, &out_res);
    if (ret != 0)
    {
        printf("inference_cn_clip_model fail! ret=%d\n", ret);
        goto out;
    }

    printf("--> rknn CN_CLIP demo result \n");
    printf("images: %s  \n", img_path);
    printf("text  : %s  \n", input_texts[out_res.text_index]);
    printf("score : %.3f\n", out_res.score);

out:
    ret = release_cn_clip_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_clip_model fail! ret=%d\n", ret);
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
