import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from yolov5.utils.plots import plot_one_box
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
import sys
sys.path.insert(0, './yolov5')

import cv2


class ObjectDetector:

    def __init__(self, weight, device):
        
        self.weight = weight
        
        self.device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu")   
        #load model
        self.model = attempt_load(self.weight, map_location=self.device)  # load FP32 model

    def get_yolo_roi(self, img_path):
        min_area_thr = 20*20

        dataset = LoadImages(img_path, img_size=640)

        for path, img, im0s, vid_cap in dataset:
            p, s, im0 = Path(path), '', im0s

            img = torch.from_numpy(img).to(self.device)
            img = img.float()
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img)[0]

            # Apply NMS
            pred = non_max_suppression(pred, 0.25, 0.45)

            bboxs = []
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class

                    # results
                    bboxs = [] 
                    for *xyxy, conf, cls in reversed(det):
                        box = [int(x.cpu().item()) for x in xyxy]
                        if (box[3]-box[1]+1)*(box[2]-box[0]+1) > min_area_thr:
                            bboxs.append( tuple(box) )

            return bboxs

    def delCoverBboxes(self, bboxes):
        cover_thr = 0.5

        xmin = np.array([bbox[0] for bbox in bboxes])
        ymin = np.array([bbox[1] for bbox in bboxes])
        xmax = np.array([bbox[2] for bbox in bboxes])
        ymax = np.array([bbox[3] for bbox in bboxes])
        bbox_areas = (ymax-ymin+1) * (xmax-xmin+1)

        sort_idx = bbox_areas.argsort()#Index of bboxes sorted in ascending order by area size
        
        keep_idx = []
        for i in range(sort_idx.size):
            #Calculate the point coordinates of the intersection
            x11 = np.maximum(xmin[sort_idx[i]], xmin[sort_idx[i+1:]]) 
            y11 = np.maximum(ymin[sort_idx[i]], ymin[sort_idx[i+1:]])
            x22 = np.minimum(xmax[sort_idx[i]], xmax[sort_idx[i+1:]])
            y22 = np.minimum(ymax[sort_idx[i]], ymax[sort_idx[i+1:]])
            #Calculate the intersection area
            w = np.maximum(0, x22-x11+1)    
            h = np.maximum(0, y22-y11+1)  
            overlaps = w * h
            
            ratios = overlaps / bbox_areas[sort_idx[i]]
            num = ratios[ratios > cover_thr]
            if num.size == 0:  
                keep_idx.append(sort_idx[i])

        return [bboxes[i] for i in keep_idx]

    @staticmethod
    def draw_bbox(img, b_box, color, width):
        for box in b_box:
            (xmin, ymin, xmax, ymax) = box 
            print(box)
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax), color,width)
        return img

    @staticmethod
    def overlap_area(box1, box2):
        (xmin, ymin, xmax, ymax) = box1
        (amin, bmin, amax, bmax) = box2
        if (xmax <= amin or amax<=xmin) and (ymax<=bmin or bmax<=ymin):
            return 0
        else:
            len = min(xmax, amax) - max(xmin, amin)
            wid = min(ymax, bmax) - max(ymin, bmin)
            return len*wid

    @staticmethod
    def area(box):
        return abs(box[2]-box[0])*(box[3]-box[1])
        
    def getRoI(self, frames):
        if type(frames) == list:
            yolo_boxes = self.get_yolo_roi(frames[-1])
        else:
            yolo_boxes = self.get_yolo_roi(frames)

        if len(yolo_boxes) > 1:
            yolo_boxes = self.delCoverBboxes(yolo_boxes)
        else:
            pass    

        return yolo_boxes

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", default='yolov5/weights/yolov5s.pt')
    parser.add_argument("--datadir", default="../VAD_datasets/ShanghaiTech/training/frames")
    parser.add_argument("--datatype", default="ShanghaiTech")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--save_path", default="./bboxes/ShanghaiTech/train/")
    args = parser.parse_args()
    

    object_detector = ObjectDetector(args.weight, args.gpu)    

    frame_path = args.datadir # frame path
    clips = sorted(os.listdir(frame_path))
    print(clips)

    for clip in clips:
        path = os.path.join(frame_path,clip)
        filenames = sorted(os.listdir(path))  
        save_file = os.path.join(args.save_path, str(clip)+".npy")
        clips_roi = []
        #读取图片开始预测
        for index in range(len(filenames)):       
            img1 = os.path.join(path, filenames[index])
            roi = object_detector.getRoI(img1)

            # result = object_detector.draw_bbox(cv2.imread(img1), roi, (255,255,0), 2)
            # cv2.imshow("roi", result)
            # if cv2.waitKey(10) == 27:
            #     sys.exit()
                            
            clips_roi.append(roi)

        # 保存
        np.save(save_file, clips_roi)
        print("save {}".format(clip))

