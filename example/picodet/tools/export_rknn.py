#!/usr/bin/env python
# coding: utf-8
import os
import sys
from rknn.api import RKNN

speech_length = 100

DEFAULT_QUANT = False

def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)]".format(sys.argv[0]))
        print("       platform choose from [rk3566, rk3568, rk3576, rk3588]")
        print("       dtype choose from [fp] for [rk3566, rk3568, rk3576, rk3588]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ['i8', 'u8', 'fp']:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        elif model_type in ['i8', 'u8']:
            do_quant = True
        else:
            do_quant = False

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = model_path.replace('.onnx', '.rknn')

    return model_path, platform, do_quant, output_path

if __name__ == '__main__':
    model_path, platform, do_quant, output_path = parse_arg()

    # Create RKNN object
    #rknn = RKNN(verbose=True)
    rknn = RKNN(verbose=False)

    print('--> Config model')
    rknn.config(target_platform=platform, mean_values=[[123.675,116.28,103.53]], std_values=[[58.395,57.12,57.375]])
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(
        model=model_path,
        inputs=["image", "scale_factor"],
        outputs = ["Div.1", "Concat.9"],
        input_size_list=[[1, 3, 416, 416], [1,2]],
    )
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')
    
    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()
