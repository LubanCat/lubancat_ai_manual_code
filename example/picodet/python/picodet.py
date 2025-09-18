#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import platform
import argparse
import cv2
import os
import time
import numpy as np
from rknnlite.api.rknn_lite import RKNNLite

IMG_SIZE = (416, 416)  # (width, height)

SCORE_THRESH = 0.5
NMS_THRESH = 0.5

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

class PicoDet:
    def __init__(
        self,
        model_path: str = "./model.rknn",
        target: str = "rk3588",
    ):
        """
        Args:
        model_path:
            rknn model path.
        target:
            device, e.g., RK3566_RK3568, RK3562, RK3576, RK3588.
        """

        self.path = model_path
        self.target = target
        self.model = self.init_model()

    def init_model(self):
        # Create RKNN object
        rknn = RKNNLite()

        # Load model
        ret = rknn.load_rknn(self.path)
        if ret != 0:
            print('Load {} failed!'.format(self.path))
            exit(ret)

        # init runtime environment
        # Run on RK356x / RK3576 / RK3588 with Debian OS, do not need specify target.
        if self.target in ['rk3576', 'rk3588']:
            # For RK3576 / RK3588, specify which NPU core the model runs on through the core_mask parameter.
            ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        else:
            ret = rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        
        return rknn

    def run(self, input_img, scale_factors):
        # set inputs
        inputs = [input_img, scale_factors]

        # Inference
        outputs = self.model.inference(inputs=inputs)
        
        # postprocess and return results
        return self.postprocess(outputs)

    def postprocess(self, outputs):
        
        batch_size, _, _ = outputs[0].shape  # (1, n, 4)
        assert batch_size == 1, "only support batch size 1."

        class_num = outputs[1].shape[1]  # (1, class_num, n)
        assert class_num == len(CLASSES), "class number should be {}.".format(len(CLASSES))
        
        boxes = outputs[0].squeeze(0)
        scores = outputs[1].squeeze(0)

        # nms
        nboxes, nclasses, nscores = [], [], []
        for i in range(len(CLASSES)):
            index = np.where(scores[i, :] > SCORE_THRESH)[0]
            if index.size == 0:
                continue
            b = boxes[index,:]
            s = scores[i,index]
            keep = nms_boxes(b, s)
            
            if len(keep) != 0:
                nboxes.append(b[keep])
                nscores.append(s[keep])
                nclasses.append(np.array([i for _ in range(len(keep))]))

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Picedet rknn demo.')
    parser.add_argument('--model_path', type=str, required= True, help='model path, could be .pt or .rknn file')
    parser.add_argument('--target', type=str, default='rk3588', help='target RKNPU platform')
    parser.add_argument('--img_path', type=str, default='../model/bus.jpg', help='img path')
    
    args = parser.parse_args()
    
    # input data proprocess
    img_src = cv2.imread(args.img_path)
    if img_src is None:
        print("imread img failed, please check: {}".format(args.img_path))
        exit(-1)

    h, w, c = img_src.shape
    input_img = cv2.resize(img_src, IMG_SIZE)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = np.expand_dims(input_img, axis=0)
    
    scale_factors = np.array([IMG_SIZE[1]/h , IMG_SIZE[0]/w], dtype=np.float32)
    scale_factors = np.expand_dims(scale_factors, axis=0)
    
    # init model
    model = PicoDet(model_path=args.model_path, target=args.target)
    
    # predict
    start_time = time.time()
    boxes, classes, scores = model.run(input_img, scale_factors)
    end_time = time.time()
    print(f"model predict: {(end_time - start_time)*1000} ms")
    
    # results
    if boxes is None and classes is None and scores is None:
        print("No object detected.")
    else:
        draw(img_src, boxes, scores, classes)
        cv2.imwrite("result.jpg", img_src)
        print("result saved to result.jpg")
    
    model.model.release()