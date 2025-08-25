#include "mobileclip_demo.h"

static void print_tensor_attr(const rknn_tensor_attr& attr) {

    std::string dims_str = "[";
    for (uint32_t i = 0; i < attr.n_dims; ++i) {
        dims_str += std::to_string(attr.dims[i]);
        if (i != attr.n_dims - 1) dims_str += ", ";
    }
    dims_str += "]";
 
    std::cout << "index=" << attr.index
              << ", name=" << attr.name
              << ", n_dims=" << attr.n_dims
              << ", dims=" << dims_str
              << ", n_elems=" << attr.n_elems
              << ", size=" << attr.size
              << ", fmt=" << get_format_string(attr.fmt)
              << ", type=" << get_type_string(attr.type)
              << ", qnt_type=" << get_qnt_type_string(attr.qnt_type)
              << ", zp=" << attr.zp
              << ", scale=" << attr.scale
              << std::endl;
}

MobileClip::MobileClip(const MobileClipModelConfig & config)
{
    debug = config.debug;
    image_size = config.image_size;

    // init rknn
    init_mobileclip_model(&encode_image_ctx, config.img_model_path);
    init_mobileclip_model(&encode_text_ctx, config.text_model_path);
}

MobileClip::~MobileClip() {
    release_clip_model();
}

void MobileClip::init_mobileclip_model(rknn_clip_context* clip_ctx, const std::string &model_path)
{
    int ret;
    rknn_context ctx = 0;
    ret = rknn_init(&ctx, (char*)model_path.c_str(), 0, 0, NULL);
    if (ret < 0)
    {
        std::cout<< "rknn_init fail ret=" << ret << std::endl;
        exit(-1);
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        std::cout<<"rknn_query fail! ret=" << ret << std::endl;
        exit(-1);
    }

    if (debug)
    {
        std::cout<<"model input num: " << io_num.n_input << ", output num: " << io_num.n_output << std::endl;
    }

    // Get Model Input Info
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            std::cout<< "rknn_query fail! ret=" << ret << std::endl;
            exit(-1);
        }
        if (debug)
        {
            std::cout << "Input Tensor " << i << ": ";
            print_tensor_attr(input_attrs[i]);
        }
    }

    // Get Model Output Info
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            std::cout<< "rknn_query fail! ret=" << ret << std::endl;
            exit(-1);
        }
        if(debug)
        {
            std::cout << "Output Tensor " << i << ": ";
            print_tensor_attr(output_attrs[i]);
        }
    }

    // Set to context
    clip_ctx->rknn_ctx = ctx;
    clip_ctx->io_num = io_num;
    clip_ctx->input_attrs.resize(io_num.n_input);
    clip_ctx->output_attrs.resize(io_num.n_output);
    clip_ctx->input_attrs = std::vector<rknn_tensor_attr>(input_attrs, input_attrs + io_num.n_input);
    clip_ctx->output_attrs = std::vector<rknn_tensor_attr>(output_attrs, output_attrs + io_num.n_output);

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        clip_ctx->model_channel = input_attrs[0].dims[1];
        clip_ctx->model_height = input_attrs[0].dims[2];
        clip_ctx->model_width = input_attrs[0].dims[3];

        if(debug)
        {
            std::cout << "Model Input Format: NCHW " << " Height: " << clip_ctx->model_height;
            std::cout << " Width: " << clip_ctx->model_width;
            std::cout << " Channel: " << clip_ctx->model_channel << std::endl;
        }
    }
    else if (input_attrs[0].fmt == RKNN_TENSOR_NHWC)
    {
        clip_ctx->model_height = input_attrs[0].dims[1];
        clip_ctx->model_width = input_attrs[0].dims[2];
        clip_ctx->model_channel = input_attrs[0].dims[3];

        if(debug)
        {
            std::cout << "Model Input Format: NHWC " << " Height: " << clip_ctx->model_height;
            std::cout << " Width: " << clip_ctx->model_width;
            std::cout << " Channel: " << clip_ctx->model_channel << std::endl;
        }
    }
    else
    {
        clip_ctx->model_height = input_attrs[0].dims[0];
        clip_ctx->model_width = input_attrs[0].dims[1];
        if(debug)
        {
            std::cout << "Model Input Format: UNDEFINED" << " Batch Size: " << clip_ctx->model_height;
            std::cout << " Sequence Length: " << clip_ctx->model_width << std::endl;
        }
    }
}

void MobileClip::release_clip_model()
{
    if (encode_image_ctx.rknn_ctx != 0)
    {
        rknn_destroy(encode_image_ctx.rknn_ctx);
        encode_image_ctx.rknn_ctx = 0;
    }

    if (encode_text_ctx.rknn_ctx != 0)
    {
        rknn_destroy(encode_text_ctx.rknn_ctx);
        encode_text_ctx.rknn_ctx = 0;
    }
}

// 保持宽高比的缩放
cv::Mat preprocess(const cv::Mat& input, int resize_size) {
    // 计算缩放比例
    float ratio = resize_size / (float)std::max(input.cols, input.rows);

    // 计算新尺寸
    int new_w = cvRound(input.cols * ratio);
    int new_h = cvRound(input.rows * ratio);
    
    // 缩放图像
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    
    // 目标尺寸图像，并填充黑色
    cv::Mat padded = cv::Mat::zeros(resize_size, resize_size, input.type());
    
    // 计算填充位置（居中）
    int x = (resize_size - new_w) / 2;
    int y = (resize_size - new_h) / 2;
    
    // 复制图像到画布
    resized.copyTo(padded(cv::Rect(x, y, new_w, new_h)));
    return padded;
}

cv::Mat MobileClip::mobileclip_image_infer(cv::Mat img)
{
    int ret;
    cv::Mat dst_img;
    rknn_input inputs[1];
    rknn_output outputs[1];

    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    if (img.channels() == 4) {
        cv::cvtColor(img, img, cv::COLOR_BGRA2BGR);
    }

    dst_img = preprocess(img, image_size);
    cv::imwrite("./preprocess.jpg", dst_img);
    cv::cvtColor(dst_img, dst_img, cv::COLOR_BGR2RGB);

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = encode_image_ctx.model_width * encode_image_ctx.model_height * encode_image_ctx.model_channel;
    inputs[0].buf = dst_img.data;

    ret = rknn_inputs_set(encode_image_ctx.rknn_ctx, encode_image_ctx.io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        exit(-1);
    }

    // Run
    ret = rknn_run(encode_image_ctx.rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        exit(-1);
    }

    // Get Output
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(encode_image_ctx.rknn_ctx, 1, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        exit(-1);
    }

    cv::Mat result(1, 512, CV_32F);
    memcpy(result.data, (float*)outputs[0].buf, outputs[0].size);

    cv::normalize(result, result);

    // Remeber to release rknn output
    rknn_outputs_release(encode_image_ctx.rknn_ctx, 1, outputs);

    return result;
}

cv::Mat MobileClip::mobileclip_text_infer(std::vector<int> tokens)
{
    int ret;
    rknn_input inputs[1];
    rknn_output outputs[1];

    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_INT32;
    inputs[0].fmt = RKNN_TENSOR_UNDEFINED;
    inputs[0].size = encode_text_ctx.model_width * encode_text_ctx.model_height * sizeof(int32_t);
    inputs[0].buf = tokens.data();

    ret = rknn_inputs_set(encode_text_ctx.rknn_ctx, 1, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        exit(-1);
    }

    // Run
    ret = rknn_run(encode_text_ctx.rknn_ctx, NULL);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        exit(-1);
    }

    // Get Output
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(encode_text_ctx.rknn_ctx, 1, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        exit(-1);
    }

    cv::Mat result(1, 512, CV_32F);
    memcpy(result.data, (float*)outputs[0].buf, outputs[0].size);

    cv::normalize(result, result);

    // Remeber to release rknn output
    rknn_outputs_release(encode_text_ctx.rknn_ctx, 1, outputs);

    return result;
}