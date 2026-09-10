import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

# Model from https://github.com/airockchip/rknn_model_zoo
ONNX_MODEL = 'yolov5s_relu.onnx'
RKNN_MODEL = 'yolov5s_relu.rknn'
IMG_PATH = './bus.jpg'
DATASET = './dataset.txt'

QUANTIZE_ON = True

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Accuracy analysis
    print('--> Accuracy analysis')
    ret = rknn.accuracy_analysis(inputs=[IMG_PATH], output_dir='./snapshot', target='rk3588',device_id='192.168.103.150:5555')
    if ret != 0:
        print('Accuracy analysis failed!')
        exit(ret)
    print('done')

    rknn.release()

