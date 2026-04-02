import numpy as np
import cv2
from copy import copy
import platform
from rknnlite.api import RKNNLite
import time 

# decice tree for RK356x/RK3576/RK3588
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'

def get_host():
    # get platform and device type
    system = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine
    if os_machine == 'Linux-aarch64':
        try:
            with open(DEVICE_COMPATIBLE_NODE) as f:
                device_compatible_str = f.read()
                if 'rk3562' in device_compatible_str:
                    host = 'RK3562'
                elif 'rk3576' in device_compatible_str:
                    host = 'RK3576'
                elif 'rk3588' in device_compatible_str:
                    host = 'RK3588'
                else:
                    host = 'RK3566_RK3568'
        except IOError:
            print('Read device node {} failed.'.format(DEVICE_COMPATIBLE_NODE))
            exit(-1)
    else:
        host = os_machine
    return host

# RKNN模型路径
# lubancat-0/1/2系列 对应 RK3566/RK3568
# lubancat-3/4/5系列 分别对应 RK3576/RK3588
RK3588_RKNN_MODEL = 'yolo26n.rknn'
RK3576_RKNN_MODEL = 'yolo26n_for_rk3576.rknn'
RK3566_RK3568_RKNN_MODEL = 'yolo26n_for_rk3566_rk3568.rknn'
RK3562_RKNN_MODEL = 'yolo26n_for_rk3562.rknn'

IMG_PATH = '../model/bus.jpg'
OBJ_THRESH = 0.25
IMG_SIZE = 640

CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")

def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    print("{:^12} {:^12}  {}".format('class', 'score', 'xmin, ymin, xmax, ymax'))
    print('-' * 50)
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[int(cl)], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

        print("{:^12} {:^12.3f} [{:>4}, {:>4}, {:>4}, {:>4}]".format(CLASSES[int(cl)], score, top, left, right, bottom))

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r  # ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def get_real_box(src_shape, box, dw, dh, ratio):
    bbox = copy(box)
    # unletter_box result
    bbox[:,0] -= dw
    bbox[:,0] /= ratio
    bbox[:,0] = np.clip(bbox[:,0], 0, src_shape[1])

    bbox[:,1] -= dh
    bbox[:,1] /= ratio
    bbox[:,1] = np.clip(bbox[:,1], 0, src_shape[0])

    bbox[:,2] -= dw
    bbox[:,2] /= ratio
    bbox[:,2] = np.clip(bbox[:,2], 0, src_shape[1])

    bbox[:,3] -= dh
    bbox[:,3] /= ratio
    bbox[:,3] = np.clip(bbox[:,3], 0, src_shape[0])
    return bbox

def postprocess_yolo26(outputs):
    """
    后处理 - 三尺度输出解码
    
    参数:
        outputs: (1, 84, 80, 80), (1, 84, 40, 40), (1, 84, 20, 20)
    
    返回:
        boxes: (N, 4) - [x1, y1, x2, y2] 归一化到原图
        scores: (N,) - 置信度
        classes: (N,) - 类别索引
    """

    all_boxes, all_scores, all_classes = [], [], []
    
    # strides for 3 scales
    strides = [8, 16, 32]
    
    for i, output in enumerate(outputs):
        # output shape: (1, 84, h, w) -> (84, h*w)
        pred = output[0].reshape(84, -1)
        
        h, w = output.shape[2], output.shape[3]
        stride = strides[i]
        
        # anchor_points
        y = np.arange(h) * stride + stride // 2
        x = np.arange(w) * stride + stride // 2
        xx, yy = np.meshgrid(x, y)
        anchor_points = np.stack([xx.ravel(), yy.ravel()], axis=0)  # (2, N)
        
        #box cls_scores
        box_dist = pred[:4, :]  # (4, N)
        cls_scores = pred[4:, :]  # (80, N)
        
        # dist2bbox
        x1y1 = anchor_points - box_dist[:2, :] * stride
        x2y2 = anchor_points + box_dist[2:, :] * stride
        boxes = np.concatenate([x1y1, x2y2], axis=0)  # (4, N)
        
        # max_cls_scores
        max_cls_scores = cls_scores.max(axis=0)  # (N,)
        
        mask = max_cls_scores > OBJ_THRESH
        if not mask.any():
            continue
        
        # classes
        classes = cls_scores.argmax(axis=0)

        all_boxes.append(boxes[:, mask])
        all_scores.append(max_cls_scores[mask])
        all_classes.append(classes[mask])
    
    if not all_boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0)
    
    boxes = np.concatenate(all_boxes, axis=1).T  # (N, 4)
    scores = np.concatenate(all_scores)
    classes = np.concatenate(all_classes)

    return boxes, scores, classes

if __name__ == '__main__':

    # Get device information
    host_name = get_host()
    if host_name == 'RK3566_RK3568':
        rknn_model = RK3566_RK3568_RKNN_MODEL
    elif host_name == 'RK3562':
        rknn_model = RK3562_RKNN_MODEL
    elif host_name == 'RK3576':
        rknn_model = RK3576_RKNN_MODEL
    elif host_name == 'RK3588':
        rknn_model = RK3588_RKNN_MODEL
    else:
        print("This demo cannot run on the current platform: {}".format(host_name))
        exit(-1)

    rknn_lite =  RKNNLite()

    # Load RKNN model
    ret = rknn_lite.load_rknn(rknn_model)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)

    # 设置输入
    img_src = cv2.imread(IMG_PATH)
    src_shape = img_src.shape[:2]
    img, ratio, (dw, dh) = letterbox(img_src, new_shape=(IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, 0)

    # Init runtime environment
    if host_name in ['RK3576', 'RK3588']:
        # For RK3576 / RK3588, specify which NPU core the model runs on through the core_mask parameter.
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    
    # inference
    outputs = rknn_lite.inference(inputs=[img])
    inferenceTime = time.time()

    # postprocess
    input_data = [outputs[0], outputs[1], outputs[2]]
    boxes, scores, classes = postprocess_yolo26(input_data)

    
    if len(boxes) == 0:
        print('No objects detected.')
    else:
        boxes = get_real_box(src_shape, boxes, dw, dh, ratio)
        draw(img_src, boxes, scores, classes)
        cv2.imwrite('result.jpg', img_src)
        print('Save results to result.jpg!')

    rknn_lite.release()
