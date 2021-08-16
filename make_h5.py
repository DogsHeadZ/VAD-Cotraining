import cv2
import h5py
import os
import shutil
from tqdm import tqdm
import numpy as np
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

from vad_dataloader_object import *

"""
For training speed, we translate the video datasets into a single h5py file for reducing the indexing time in Disk
By keeping the compressed type as JPG, we can reduce the memory space
Here, we give the example as translating UCF-Crime training set into a single h5py file, you can modify it for other dataset
OR
You can modify the datasets/dataset.py for directly using the video files for testing!
"""

# 和VadDataset 都一样的，就是多返回了一个key
class VadDataset_h5(VadDataset):
    def __getitem__(self, index):

        video_name = os.path.split(os.path.split(self.samples[index])[0])[1]
        frame_name = int(os.path.split(self.samples[index])[1].split('.')[-2])

        if self.bbox_folder is not None:  # 已经提取好了bbox，直接读出来
            bboxes = self.videos[video_name]['bbox'][frame_name]
        else:  # 需要重新计算
            last_frame = self.videos[video_name]['frame'][frame_name]
            bboxes = self.yolo_model.getRoI(last_frame)

        if len(bboxes)==0: # 对于一些帧中没有object的情况
            print("no object in the frame")
            return None, None

        object_batch = []
        for bbox in bboxes:
            # img
            one_object_batch = []
            for i in range(self._time_step):
                image = np_load_frame_roi(self.videos[video_name]['frame'][frame_name-self._time_step+1+i], self._resize_height,
                                          self._resize_width, bbox)  # 这里的64是裁剪框的resize
                if self.transform is not None:
                    one_object_batch.append(self.transform(image))
            one_object_batch = torch.stack(one_object_batch, dim=0)
            # print("object_batch.shape: ", object_batch.shape)
            object_batch.append(one_object_batch)
        object_batch = torch.stack(object_batch, dim=0)  # 大小为[目标个数, _time_step, 图片的通道数, _resize_height, _resize_width]       


        # frame batch
        frames_flow = []
        if self.flow_folder is not None:  # 已经提取好了，直接读出来
            for i in range(0, self._time_step):
                frames_flow.append( torch.load(self.videos[video_name]['flow'][frame_name-self._time_step+1+i]) )
        else:
            for i in range(0, self._time_step):
                frame_flow = self.flownet.get_frame_flow(self.videos[video_name]['frame'][frame_name-self._time_step+1+i],
                                                self.videos[video_name]['frame'][frame_name-self._time_step+2+i], 512, 384)
                frames_flow.append(frame_flow)

        w_ratio = 512 / self.img_size[1]      # 光流图的大小是512*384
        h_ratio = 384 / self.img_size[0] 
        trans_bboxes = [[int(box[0] * w_ratio), int(box[1] * h_ratio),
                         int(box[2] * w_ratio), int(box[3] * h_ratio)] for box in bboxes] # example[(290, 91, 301, 109), (332, 94, 343, 113)]

        # print(trans_bboxes)

        # 裁剪出对应区域的光流
        flow_batch = []
        for bbox in trans_bboxes:
            # flow
            one_object_batch = []
            for i in range(self._time_step):
                flow = load_flow_roi(frames_flow[i], self._resize_height, self._resize_width, bbox)  # 这里的64是裁剪框的resize
                one_object_batch.append(flow)
            one_object_batch = torch.stack(one_object_batch, dim=0)
            # print("object_batch.shape: ", object_batch.shape)
            flow_batch.append(one_object_batch)
        flow_batch = torch.stack(flow_batch, dim=0)  # 大小为[目标个数, _time_step, 图片的通道数, _resize_height, _resize_width]

        # print(object_batch.shape, flow_batch.shape)
        key = str(video_name) + "-" + str(frame_name)
        return key, object_batch, flow_batch #rgb, flow


def Video2ImgH5(video_folder, h5_path, dataset, _time_step):
    # not multi-thread, may take time
    h_rgb = h5py.File(os.path.join(h5_path, "rgb.h5"), 'a')
    h_flow = h5py.File(os.path.join(h5_path, "flow.h5"), 'a')
    train_data = VadDataset_h5(args, video_folder=video_folder, bbox_folder=None, dataset="ShanghaiTech", flow_folder=None,
                            device="0", transform=transforms.Compose([transforms.ToTensor()]),
                            resize_height=64, resize_width=64,time_step=_time_step)
    train_loader = DataLoader(dataset=train_data, shuffle=False, batch_size=1)
    
    # key = video_name + frame_num
    # 如果这一帧没有object，则不保存
    for j, (key, rgb, flow) in enumerate(tqdm(train_loader, desc='make_h5', leave=False)):
        key = key[0]
        rgb = rgb[0]
        flow = flow[0]
        if rgb == None:
            continue
        h_rgb.create_dataset(key, data=rgb, chunks=True)
        h_flow.create_dataset(key, data=flow.cpu(), chunks=True)        
    
    print('finished!')

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # for flownet2, no need to modify
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    video_dir='../VAD_datasets/ShanghaiTech/training/frames'
    h5_file_path='./data/ShanghaiTech-16/'
    if os.path.exists(h5_file_path):
        shutil.rmtree(h5_file_path)
    os.mkdir(h5_file_path)

    Video2ImgH5(video_dir,h5_file_path,"ShanghaiTech",_time_step=16)