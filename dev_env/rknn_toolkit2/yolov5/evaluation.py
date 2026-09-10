from rknn.api import RKNN

RKNN_MODEL = 'yolov5s_relu.rknn'
IMG_PATH = './bus.jpg'
DATASET = './dataset.txt'

if __name__ == '__main__':
    # 创建RKNN
    # 如果测试遇到问题，请开启verbose=True，查看调试信息。
    #rknn = RKNN(verbose=True)
    rknn = RKNN()
    
    # 导入RKNN模型，path参数指定模型路径
    print('--> Loading model')
    ret = rknn.load_rknn(path=RKNN_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # 初始化运行时环境，指定连接的板卡NPU平台,
    # perf_debug开启进行性能评估时开启debug模式，eval_mem进入内存评估模式
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rk3588', device_id='192.168.103.150:5555', perf_debug=True, eval_mem=True)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # 模型性能进行评估，默认is_print是true，打印内存使用情况
    print('--> eval_perf')
    rknn.eval_perf()
    print('done')

    # 调试，模型性能进行评估,默认is_print是true，打印内存使用情况
    print('--> eval_memory')
    rknn.eval_memory()
    print('done')
    
    rknn.release()

