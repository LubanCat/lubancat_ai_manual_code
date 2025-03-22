// Copyright (c) 2023 by Embedfire 

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <rknn_api.h>

void findMaxAndPosition(float* buffer, int size) {
    int maxPosition = 0;
    float maxValue = buffer[0];
    for (int i = 1; i < size; ++i) {
      if (buffer[i] > maxValue) {
          maxValue = buffer[i];
          maxPosition = i;
      }
    }
    printf("本次预测的数字是: %d \n" , maxPosition);
}

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
  for (int i = 1; i < attr->n_dims; ++i) {
    shape_str += ", " + std::to_string(attr->dims[i]);
  }

  printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
         "type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
         attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz)
{
  unsigned char* data;
  int            ret;

  data = NULL;

  if (NULL == fp) {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0) {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char*)malloc(sz);
  if (data == NULL) {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char* load_model(const char* filename, int* model_size)
{
  FILE*          fp;
  unsigned char* data;

  fp = fopen(filename, "rb");
  if (NULL == fp) {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char** argv)
{
  char*          model_name = NULL;
  rknn_context   ctx;
  int            img_width          = 0;
  int            img_height         = 0;
  int            ret;
  struct timeval start_time, stop_time;
  
  if (argc != 3) {
    printf("Usage: %s <rknn model> <jpg> \n", argv[0]);
    return -1;
  }
  
  model_name       = (char*)argv[1];
  char* image_name = argv[2];

  // Load RKNN Model
  printf("Loading mode %s ...\n", model_name);
  int            model_data_size = 0;
  unsigned char* model_data      = load_model(model_name, &model_data_size);
  ret                            = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
  if (ret < 0) {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }
  
  // Get sdk version and driver version
  rknn_sdk_version version;
  ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
  if (ret < 0) {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }
  printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

  // Get Model Input Output Info
  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret < 0) {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }
  printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, sizeof(input_attrs));
  for (int i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    ret                  = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret < 0) {
      printf("rknn_init error ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&(input_attrs[i]));
  }

  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, sizeof(output_attrs));
  for (int i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    dump_tensor_attr(&(output_attrs[i]));
  }

  int channel = 0;
  int width   = 0;
  int height  = 0;
  if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
    printf("model is NCHW input fmt\n");
    channel = input_attrs[0].dims[1];
    height  = input_attrs[0].dims[2];
    width   = input_attrs[0].dims[3];
  } else {
    printf("model is NHWC input fmt\n");
    height  = input_attrs[0].dims[1];
    width   = input_attrs[0].dims[2];
    channel = input_attrs[0].dims[3];
  }

  printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);
  
  // Set Input Data
  rknn_input inputs[1];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index        = 0;
  inputs[0].type         = RKNN_TENSOR_UINT8;
  inputs[0].size         = width * height * channel;
  inputs[0].fmt          = RKNN_TENSOR_NHWC;
  inputs[0].pass_through = 0;

  // Load image
  printf("Read %s ...\n", image_name);
  cv::Mat orig_img = cv::imread(image_name, 1);
  if (!orig_img.data) {
    printf("cv::imread %s fail!\n", image_name);
    return -1;
  }

  cv::Mat img;
  cv::cvtColor(orig_img, img, cv::COLOR_RGB2GRAY);
  img_width  = img.cols;
  img_height = img.rows;
  printf("img width = %d, img height = %d\n", img_width, img_height);

  // resize with Opencv
  if (img_width != width || img_height != height) {
    printf("resize with Opencv! \n");
    cv::Mat resize_image;
    cv::resize(img, resize_image, cv::Size(width, height), cv::INTER_LINEAR);
    inputs[0].buf = (void*)resize_image.data;
  } else {
    inputs[0].buf = (void*)img.data;
  }

  rknn_inputs_set(ctx, io_num.n_input, inputs);
  rknn_output outputs[1];
  memset(outputs,0,sizeof(outputs));
  outputs[0].want_float = 1;

  // Run
  gettimeofday(&start_time, NULL);
  ret = rknn_run(ctx, NULL);
  ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
  gettimeofday(&stop_time, NULL);
  printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

  // Get Output and Post Process
  float*   buffer = (float*)outputs[0].buf;
  printf("outputs: ");
  for (int i = 0; i < 10; i++) {
    printf("%.6f ", buffer[i]);
  }
  printf("\n");

  findMaxAndPosition(buffer, 10);

  ret = rknn_destroy(ctx);

  if (model_data) {
    free(model_data);
  }

  return 0;
}

