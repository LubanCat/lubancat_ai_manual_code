import os
from rkllm.api import RKLLM
import argparse

os.environ['CUDA_VISIBLE_DEVICES']='0'

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--modelpath', type=str, default='Qwen-1_8B-Chat', help='model path', required=True)
    argparse.add_argument('--target-platform', type=str, default='rk3588', help='target platform', required=False)
    argparse.add_argument("--dataset_path", type=str, help="calibration data path(./data_quant.json)")
    argparse.add_argument('--num_npu_core', type=int, default=3, help='npu core num(rk3588:0-3, rk3576:0-2)', required=False)
    argparse.add_argument('--optimization_level', type=int, default=1, help='optimization_level(0 or 1)', required=False)
    argparse.add_argument('--quantized_dtype', type=str, default='w8a8', help='quantized dtype(rk3588:w8a8/w8a8_g128/w8a8_g256/w8a8_g512....)', required=False)
    argparse.add_argument('--quantized_algorithm', type=str, default='normal', help='quantized algorithm(normal/grq/gdq)', required=False)
    argparse.add_argument('--device', type=str, default='cpu', help='device(cpu/cuda)', required=False)
    argparse.add_argument('--savepath', type=str, default='Qwen-1_8B-Chat.rkllm', help='save path', required=False)
    args = argparse.parse_args()

    # Different quantization methods are optimized for different algorithms:  
    # w8a8/w8a8_gx   is recommended to use the normal algorithm.  
    # w4a16/w4a16_gx is recommended to use the grq algorithm.

    qparams = None # Use extra_qparams

    # init 
    llm = RKLLM()

    # Load model
    ret = llm.load_huggingface(model=args.modelpath, model_lora = None, device=args.device, dtype="float32", custom_config=None, load_weight=True)
    # ret = llm.load_gguf(model = args.modelpath)
    if ret != 0:
        print('Load model failed!')
        exit(ret)

    # Build model
    ret = llm.build(do_quantization=True, optimization_level=args.optimization_level, quantized_dtype=args.quantized_dtype,
                    quantized_algorithm=args.quantized_algorithm, target_platform=args.target_platform, 
                    num_npu_core=args.num_npu_core, extra_qparams=qparams, dataset=args.dataset_path)
    if ret != 0:
        print('Build model failed!')
        exit(ret)

    # Export rkllm model
    ret = llm.export_rkllm(args.savepath)
    if ret != 0:
        print('Export model failed!')
        exit(ret)

