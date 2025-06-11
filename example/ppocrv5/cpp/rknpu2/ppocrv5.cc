#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <thread>

#include "ppocrv5.h"

static unsigned char* load_model(const char* filename, int* model_size)
{
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char* model = (unsigned char*)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", filename);
        free(model);
        fclose(fp);
        return NULL;
    }
    *model_size = model_len;
    fclose(fp);
    return model;
}

static void dump_input_dynamic_range(rknn_input_range *dyn_range)
{
    std::string range_str = "";
    for (int n = 0; n < dyn_range->shape_number; ++n)
    {
        range_str += n == 0 ? "[" : ",[";
        range_str += dyn_range->n_dims < 1 ? "" : std::to_string(dyn_range->dyn_range[n][0]);
        for (int i = 1; i < dyn_range->n_dims; ++i)
        {
            range_str += ", " + std::to_string(dyn_range->dyn_range[n][i]);
        }
        range_str += "]";
    }

    printf("  index=%d, name=%s, shape_number=%d, range=[%s], fmt = %s\n", dyn_range->index, dyn_range->name,
           dyn_range->shape_number, range_str.c_str(), get_format_string(dyn_range->fmt));
}

bool CompareBox(const std::array<int, 8>& result1, const std::array<int, 8>& result2)
{
    if (result1[1] < result2[1]) 
    {
        return true;
    } else if (result1[1] == result2[1]) 
    {
        return result1[0] < result2[0];
    } else 
    {
        return false;
    }
}

void SortBoxes(std::vector<std::array<int, 8>>* boxes)
{
    std::sort(boxes->begin(), boxes->end(), CompareBox);

    if (boxes->size() == 0)
    {
        return;
    }
    
    for (int i = 0; i < boxes->size() - 1; i++) {
        for (int j = i; j >=0 ; j--){
            if (std::abs((*boxes)[j + 1][1] - (*boxes)[j][1]) < 10 && ((*boxes)[j + 1][0] < (*boxes)[j][0])) 
            {
                std::swap((*boxes)[i], (*boxes)[i + 1]);
            }
        }
    }

}

cv::Mat GetRotateCropImage(const cv::Mat& srcimage, const std::array<int, 8>& box)
{
    cv::Mat image;
    srcimage.copyTo(image);

    std::vector<std::vector<int>> points;

    for (int i = 0; i < 4; ++i) {
        std::vector<int> tmp;
        tmp.push_back(box[2 * i]);
        tmp.push_back(box[2 * i + 1]);
        points.push_back(tmp);
    }
    int x_collect[4] = {box[0], box[2], box[4], box[6]};
    int y_collect[4] = {box[1], box[3], box[5], box[7]};
    int left = int(*std::min_element(x_collect, x_collect + 4));
    int right = int(*std::max_element(x_collect, x_collect + 4));
    int top = int(*std::min_element(y_collect, y_collect + 4));
    int bottom = int(*std::max_element(y_collect, y_collect + 4));

    cv::Mat img_crop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

    for (int i = 0; i < points.size(); i++) {
        points[i][0] -= left;
        points[i][1] -= top;
    }

    int img_crop_width = int(sqrt(pow(points[0][0] - points[1][0], 2) +
                                    pow(points[0][1] - points[1][1], 2)));
    int img_crop_height = int(sqrt(pow(points[0][0] - points[3][0], 2) +
                                    pow(points[0][1] - points[3][1], 2)));

    cv::Point2f pts_std[4];
    pts_std[0] = cv::Point2f(0., 0.);
    pts_std[1] = cv::Point2f(img_crop_width, 0.);
    pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
    pts_std[3] = cv::Point2f(0.f, img_crop_height);

    cv::Point2f pointsf[4];
    pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
    pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
    pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
    pointsf[3] = cv::Point2f(points[3][0], points[3][1]);

    cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

    cv::Mat dst_img;
    cv::warpPerspective(img_crop, dst_img, M,
                        cv::Size(img_crop_width, img_crop_height),
                        cv::BORDER_REPLICATE);

    if (float(dst_img.rows) >= float(dst_img.cols) * 1.5) {
        cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
        cv::transpose(dst_img, srcCopy);
        cv::flip(srcCopy, srcCopy, 0);
        return srcCopy;
    } else {
        return dst_img;
    }
}

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
            "zp=%d, scale=%f\n",
            attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
            attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
            get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

int init_ppocr_model(const char* model_path, rknn_app_context_t* app_ctx)
{
    int ret;
    int model_len = 0;
    // char* model;
    rknn_context ctx = 0;

    // Load RKNN Model
    unsigned char* model = load_model(model_path, &model_len);
    if (model == NULL)
    {
        printf("load_model fail!\n");
        return -1;
    }

    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->status = 1;
    app_ctx->input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height  = input_attrs[0].dims[2];
        app_ctx->model_width   = input_attrs[0].dims[3];
    } else {
        printf("model is NHWC input fmt\n");
        app_ctx->model_height  = input_attrs[0].dims[1];
        app_ctx->model_width   = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
        app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}


int init_ppocr_rec_model(const char* model_path, rknn_app_context_t* app_ctx)
{
    int ret;
    int model_len = 0;
    // char* model;
    rknn_context ctx = 0;

    // Load RKNN Model
    unsigned char* model = load_model(model_path, &model_len);
    if (model == NULL)
    {
        printf("load_model fail!\n");
        return -1;
    }

    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // 查询模型支持的输入形状
    printf("dynamic inputs shape range:\n");
    rknn_input_range shape_range[io_num.n_input];
    memset(shape_range, 0, io_num.n_input * sizeof(rknn_input_range));
    for (uint32_t i = 0; i < io_num.n_input; i++)
    {
        shape_range[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_DYNAMIC_RANGE, &shape_range[i], sizeof(rknn_input_range));
        if (ret != RKNN_SUCC)
        {
            fprintf(stderr, "rknn_query error! ret=%d\n", ret);
            return -1;
        }
        dump_input_dynamic_range(&shape_range[i]);
    }

    // Set to context
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->status = 1;
    app_ctx->input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height  = input_attrs[0].dims[2];
        app_ctx->model_width   = input_attrs[0].dims[3];
    } else {
        printf("model is NHWC input fmt\n");
        app_ctx->model_height  = input_attrs[0].dims[1];
        app_ctx->model_width   = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
        app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}


int release_ppocr_model(rknn_app_context_t* app_ctx)
{
    if (app_ctx->input_attrs != NULL) {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL) {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0) {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_ppocr_det_model(rknn_app_context_t* app_ctx, image_buffer_t* src_img, ppocr_det_postprocess_params* params, ppocr_det_result* out_result)
{
    int ret;
    image_buffer_t img;
    rknn_input inputs[1];
    rknn_output outputs[1];

    memset(&img, 0, sizeof(image_buffer_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Pre Process
    img.width = app_ctx->model_width;
    img.height = app_ctx->model_height;
    img.format = IMAGE_FORMAT_RGB888;
    img.size = get_image_size(&img);
    img.virt_addr = (unsigned char*)malloc(img.size);
    if (img.virt_addr == NULL) {
        printf("malloc buffer size:%d fail!\n", img.size);
        return -1;
    }

    ret = convert_image(src_img, &img, NULL, NULL, 0);
    if (ret < 0) {
        printf("convert_image fail! ret=%d\n", ret);
        return -1;
    }

    // cv::Mat img_M = cv::Mat(img.height, img.width, CV_8UC3,(uint8_t*)img.virt_addr);
    // img_M.convertTo(img_M, CV_32FC3);

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    // inputs[0].type  = RKNN_TENSOR_FLOAT32;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    // inputs[0].size  = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel * sizeof(float);
    inputs[0].buf   = img.virt_addr;
    // inputs[0].buf = malloc(inputs[0].size);
    // memcpy(inputs[0].buf, img_M.data, inputs[0].size);

    float scale_w = (float)src_img->width / (float)img.width;
    float scale_h = (float)src_img->height / (float)img.height;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, inputs);
    if (ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    // printf("rknn_run\n");
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Get Output
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        goto out;
    }

    // Post Process
    ret = dbnet_postprocess((float*)outputs[0].buf, app_ctx->model_width, app_ctx->model_height, 
                                                params->threshold, params->box_threshold, params->use_dilate, params->db_score_mode, 
                                                params->db_unclip_ratio, params->db_box_type,
                                                scale_w, scale_h, out_result);
    
    // Remeber to release rknn output
    rknn_outputs_release(app_ctx->rknn_ctx, 1, outputs);

out:
    if (img.virt_addr != NULL) {
        free(img.virt_addr);
    }

    return ret;
}

void inference_ppocr_rec_worker_thread(int id, rknn_app_context_t* app_ctx, rknn_context& ctx, 
    const cv::Mat& in_image ,  const std::vector<std::array<int, 8>> & boxes_result, ppocr_text_recog_array_result_t* out_result, int n_threads)
{
    int ret;
    int i = id;

    rknn_input inputs[1];
    rknn_output outputs[1];
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // pre set input_attrs
    rknn_tensor_attr input_attrs[1];
    memcpy(input_attrs, app_ctx->input_attrs, 1 * sizeof(rknn_tensor_attr));
    ret = rknn_set_input_shapes(ctx, 1, input_attrs);
    assert(ret >= 0);

    // pre rknn_input
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_FLOAT32;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;

    for (; i < boxes_result.size(); i += n_threads) {
        cv::Mat crop_image = GetRotateCropImage(in_image, boxes_result[i]);

        // Pre Process
        float ratio = crop_image.cols / float(crop_image.rows);
        int resized_w;
        int imgW;
        int imgH = IMAGE_HEIGHT;

        if(crop_image.cols >= 480) {
            imgW = 640;
        } 
        else if(crop_image.cols >= 240) {
            imgW = 320;
        }
        else {
            imgW = 160;
        }

        if (std::ceil(imgH*ratio) > imgW) {
            resized_w = imgW;
        }
        else {
            resized_w = std::ceil(imgH*ratio);
        }
        cv::resize(crop_image, crop_image, cv::Size(resized_w, imgH));

        crop_image.convertTo(crop_image, CV_32FC3);
        crop_image = (crop_image - 127.5)/127.5;
        if (resized_w < imgW) {
            copyMakeBorder(crop_image, crop_image, 0, 0, 0, imgW - resized_w, cv::BORDER_CONSTANT, 0);
        }

        // set input_attrs
        if(input_attrs[0].dims[2] != imgW) {
            input_attrs[0].dims[2] = imgW;
            ret = rknn_set_input_shapes(ctx, 1, input_attrs);
            assert(ret >= 0);
        }

        // set Input Data
        inputs[0].size  = imgW * app_ctx->model_height * app_ctx->model_channel * sizeof(float);
        inputs[0].buf = malloc(inputs[0].size);
        memcpy(inputs[0].buf, crop_image.data, inputs[0].size);
        ret = rknn_inputs_set(ctx, 1, inputs);
        assert(ret >= 0);

        // Run
        ret = rknn_run(ctx, nullptr);
        assert(ret >= 0);

        // Get Output
        int out_seq_len = imgW / 8;
        outputs[0].want_float = 1;

        ret = rknn_outputs_get(ctx, 1, outputs, NULL);
        assert(ret >= 0);

        // Post Process
        std::string str_res;
        float score = 0.f;
        int argmax_idx;
        int last_index = 0;
        int count = 0;
        float max_value = 0.0f;

        float* out_data = (float*)outputs[0].buf;
        for (int n = 0; n < out_seq_len; n++) {
            float * max_idx = std::max_element(&out_data[n * MODEL_OUT_CHANNEL], &out_data[(n + 1) * MODEL_OUT_CHANNEL - 1]);
            argmax_idx = int(std::distance(&out_data[n * MODEL_OUT_CHANNEL], max_idx));
            max_value = float(*max_idx);
            if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
                score += max_value;
                count += 1;
                assert(argmax_idx <= MODEL_OUT_CHANNEL);
                str_res += ocr_dict[argmax_idx];
            }
            last_index = argmax_idx;
        }
        score /= (count + 1e-6);
        if (count == 0 || std::isnan(score)) {
            score = 0;
        }

        // copy result to out_result
        if (score > TEXT_SCORE) {
            out_result->text_result[i].box.left_top.x = boxes_result[i][0];
            out_result->text_result[i].box.left_top.y = boxes_result[i][1];
            out_result->text_result[i].box.right_top.x = boxes_result[i][2];
            out_result->text_result[i].box.right_top.y = boxes_result[i][3];
            out_result->text_result[i].box.right_bottom.x = boxes_result[i][4];
            out_result->text_result[i].box.right_bottom.y = boxes_result[i][5];
            out_result->text_result[i].box.left_bottom.x = boxes_result[i][6];
            out_result->text_result[i].box.left_bottom.y = boxes_result[i][7];
            strcpy(out_result->text_result[i].text.str, str_res.c_str());
            out_result->text_result[i].text.str_size = count;
            out_result->text_result[i].text.score = score;
        } else {
            out_result->text_result[i].text.score = 0;
        }

        // Remeber to release rknn output
        rknn_outputs_release(ctx, 1, outputs);

        // free
        if (inputs[0].buf != NULL) {
            free(inputs[0].buf);
        }
    }
}

int inference_ppocrv5_model(ppocr_system_app_context* sys_app_ctx,
    image_buffer_t* src_img, ppocr_det_postprocess_params* params, ppocr_text_recog_array_result_t* out_result)
{
    int ret;
    // TIMER timer;

    // Detect Text
    ppocr_det_result det_results;
    // timer.tik();
    ret = inference_ppocr_det_model(&sys_app_ctx->det_context, src_img, params, &det_results);
    // timer.tok();
    // timer.print_time("inference_ppocr_det_model");
    if (ret != 0) {
        printf("inference_ppocr_det_model fail! ret=%d\n", ret);
        return -1;
    }

    // Recogize Text
    out_result->count = 0;
    if (det_results.count == 0) {           // detect nothing
        return 0;
    }

    // boxes to boxes_result
    std::vector<std::array<int, 8>> boxes_result;
    for (int i=0; i < det_results.count; i++) {
        std::array<int, 8> new_box;
        new_box[0] = det_results.box[i].left_top.x;
        new_box[1] = det_results.box[i].left_top.y;
        new_box[2] = det_results.box[i].right_top.x;
        new_box[3] = det_results.box[i].right_top.y;
        new_box[4] = det_results.box[i].right_bottom.x;
        new_box[5] = det_results.box[i].right_bottom.y;
        new_box[6] = det_results.box[i].left_bottom.x;
        new_box[7] = det_results.box[i].left_bottom.y;
        boxes_result.emplace_back(new_box);
    }

    // Sort text boxes in order from top to bottom, left to right for speeding up
    SortBoxes(&boxes_result);

    // text recognize
    cv::Mat in_image = cv::Mat(src_img->height, src_img->width, CV_8UC3,(unsigned char*)src_img->virt_addr);

    out_result->count  = boxes_result.size();

    // set core mask
    // rknn_core_mask core_mask = RKNN_NPU_CORE_ALL;
    // rknn_set_core_mask(sys_app_ctx->rec_context.rknn_ctx, core_mask);

    int n_threads = std::thread::hardware_concurrency();
    rknn_context ctx[n_threads-1];
    std::vector<std::thread> workers(n_threads - 1);
    for (int iw = 0; iw < n_threads - 1; ++iw) {

        ret = rknn_dup_context(&sys_app_ctx->rec_context.rknn_ctx, &ctx[iw]);
        if (ret < 0) {
            printf("rknn_dup_context fail! ret=%d\n", ret);
            return -1;
        }

        workers[iw] = std::thread(
                inference_ppocr_rec_worker_thread, iw + 1, &sys_app_ctx->rec_context, std::ref(ctx[iw]),
                std::cref(in_image), std::cref(boxes_result), out_result, n_threads);
    }
    // main thread
    inference_ppocr_rec_worker_thread(0, &sys_app_ctx->rec_context, sys_app_ctx->rec_context.rknn_ctx, 
                in_image, boxes_result, out_result, n_threads);

    for (int iw = 0; iw < n_threads - 1; ++iw) {
        workers[iw].join();
    }

    for (int iw = 0; iw < n_threads - 1; ++iw) {
        if (ctx[iw] != 0) {
            rknn_destroy(ctx[iw]);
            ctx[iw] = 0;
        }
    }

    return ret;
}

