#ifndef _RKNN_YOLO26_DEPTH_DEMO_POSTPROCESS_H_
#define _RKNN_YOLO26_DEPTH_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>

#include "rknn_api.h"
#include "common.h"
#include "image_utils.h"

int post_process_pre(rknn_app_context_t *app_ctx, std::vector<float>& output, letterbox_t *letter_box, cv::Mat& depth_maps);

cv::Mat render_depth_overlay(const cv::Mat& bgr_orig, const cv::Mat& heat, double alpha = 0.6);

cv::Mat colorize_depth(const cv::Mat& depth, bool disparity = true, double p_lo = 2.0, double p_hi = 98.0, int cmap = cv::COLORMAP_JET);

#endif //_RKNN_YOLO26_DEPTH_DEMO_POSTPROCESS_H_