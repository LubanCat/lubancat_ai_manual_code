# 目录说明

| 目录                | 对应教程章节                       |
| ------------------ | --------------------------------- |
| picodet            | [PP-Picodet](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/picodet.html)    | 
| ppocrv4            | [PP-ORCv4](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/PP-ORCv4.html)                        | 
| ppocrv5            | [PP-ORCv5](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/PP-ocrv5.html)                        | 
| ppseg              | [PP-LitetSeg](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/ppseg.html)                     |
| ppyoloe            | [PP-YOLOE](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/ppyoloe.html)                        |
| RT-DETR            | [RT-DETR](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/RT-DETR.html)                         |
| yolox              | [YoloX](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/yolox.html)                           |
| yolov5             | [YOLOv5(目标检测)](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/yolov5.html)                 |
| yolov5_seg         | [YOLOv5(实例分割)](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/yolov5_seg.html)                 |
| yolov5face         | [YOLOv5Face(人脸检测)](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/yolov5_face.html)             |
| yolov8             | [YOLOv8](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/yolov8.html)                 |
| yolov8-obb         | [YOLOv8旋转目标检测](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/yolov8_obb.html)                |
| yolov10            | [YOLOv10](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/yolov10.html)                     |
| yolo11             | [YOLOv11](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/yolo11.html)                        |
| mobileclip         | [MobileCLIP](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/clip.html#id14)                        | 
| owl-vit            | [OWL-ViT](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/owl-vit.html)                        | 
| sense-voice        | [Sensevoice](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/snesevoice.html)                        | 
| melotts            | [MeloTTS](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/tts.html#melotts)                        |
| garbage_detection  | [垃圾检测和识别](https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/example/garbage_detect.html)               |


# 文件说明

scaling_frequency.sh 是系统CPU，DDR，NPU频率修改脚本，例如：

```sh
# USAGE: ./fixed_frequency.sh -c {chip_name} [-h]"
# "  -c:  chip_name, such as rv1126 / rk3588"
# "  -h:  Help"
sudo bash scaling_frequency.sh -c rk3568
```

# 问题反馈

如果有任何问题请联系淘宝野火官方旗舰店技术支持反馈。