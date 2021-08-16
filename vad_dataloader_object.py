import argparse
import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

import torch.nn.functional as F

from tqdm import tqdm
from getROI import ObjectDetector
from getFlow import FlowNet


def np_load_frame_roi(filename, resize_height, resize_width, bbox):
    (xmin, ymin, xmax, ymax) = bbox
    image_decoded = cv2.imread(filename)
    image_decoded = image_decoded[ymin:ymax, xmin:xmax]

    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    # image_resized = image_resized.astype(dtype=np.float32)
    # image_resized = (image_resized / 127.5) - 1.0
    return image_resized

def load_flow_roi(flow, resize_height, resize_width, bbox):
    (xmin, ymin, xmax, ymax) = bbox
    # print(flow.shape)
    
    flow = flow[:, ymin:ymax, xmin:xmax]
    flow = flow.unsqueeze(0)
    flow = F.interpolate(flow, size=([resize_width, resize_height]), mode='bilinear', align_corners=False)
    flow = flow.squeeze(0)

    return flow

def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].
    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    # image_resized = image_resized.astype(dtype=np.float32)
    # image_resized = (image_resized )/255.0

    # image_resized = (image_resized / 127.5) - 1.0
    return image_resized

def my_collate(batch):
    rgb_batch = []
    flow_batch = []
    for sample in batch:
        if sample[0] != None:
            rgb_batch.append(sample[0])
            flow_batch.append(sample[1])

    if len(rgb_batch) == 0:
        return None, None
    
    rgb_batch = pad_sequence(rgb_batch).transpose(0,1)
    flow_batch = pad_sequence(flow_batch).transpose(0,1) 
    return rgb_batch, flow_batch



class VadDataset(data.Dataset):
    def __init__(self, flowargs, video_folder, transform, resize_height, resize_width, dataset='', time_step=5,
                 bbox_folder=None, device='0', flow_folder=None):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self.dataset = dataset  # ped2 or avenue or ShanghaiTech

        self.bbox_folder = bbox_folder  # 如果box已经预处理了，则直接将npy数据读出来, 如果没有，则在get_item的时候计算
        if bbox_folder == None:  # 装载yolo模型
            self.yolo_weights = 'yolov5/weights/yolov5s.pt'
            self.yolo_device = device
            self.yolo_model = ObjectDetector(self.yolo_weights, self.yolo_device)
        

        self.flow_folder = flow_folder # 如果flow已经预处理了，则直接将npy数据读出来, 如果没有，则在get_item的时候计算
        if self.flow_folder == None:  # 装载flownet
            self.device = device
            self.flownet = FlowNet(flowargs,"flownet2/FlowNet2_checkpoint.pth.tar", device)

        self.setup()
        self.samples = self.get_all_samples()

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            # video_name = video.split('/')[-1]           #视频的目录名即类别如01, 02, 03, ...
            video_name = os.path.split(video)[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))  # 每个目录下的所有视频帧
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])  # 每个目录下视频帧的个数
        # print(self.videos.keys())
        video_name = os.path.split(videos[0])[-1]
        self.img_size = cv2.imread(self.videos[video_name]['frame'][0]).shape  # [h, w, c]
        if self.bbox_folder != None:  # 如果box已经预处理了，则直接将npy数据读出来
            for bbox_file in sorted(os.listdir(self.bbox_folder)):
                video_name = bbox_file.split('.')[0]
                self.videos[video_name]['bbox'] = np.load(os.path.join(self.bbox_folder, bbox_file),allow_pickle=True)  # 每个目录下所有视频帧预提取的bbox
                # print("video_name: {}, bboxsize: {}".format(video_name, len(self.videos[video_name]['bbox'])) )

        # TODO: Optical Flow
        if self.flow_folder != None:  # 如果已经预处理了，直接读取
            for flow_dir in sorted(os.listdir(self.flow_folder)):
                video_name = flow_dir
                path = os.path.join(self.flow_folder, flow_dir)
                self.videos[video_name]['flow'] = sorted(glob.glob(os.path.join(path, '*')))
        # print(self.videos[video_name]['flow'])

    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = os.path.split(video)[-1]
            for i in range(self._time_step-1, len(self.videos[video_name]['frame'])-1):  # 从_time_step为了刚好能够滑窗到视频尾部，减1是为了光流的计算
                frames.append(self.videos[video_name]['frame'][i])  # frames存储着训练时每段视频片段的最后帧，根据最后一帧向前进行滑窗即可得到这段视频片段

        return frames

    def __getitem__(self, index):

        video_name = os.path.split(os.path.split(self.samples[index])[0])[1]
        frame_name = int(os.path.split(self.samples[index])[1].split('.')[-2])

        if self.bbox_folder is not None:  # 已经提取好了bbox，直接读出来
            bboxes = self.videos[video_name]['bbox'][frame_name]
        else:  # 需要重新计算
            last_frame = self.videos[video_name]['frame'][frame_name]
            bboxes = self.yolo_model.getRoI(last_frame)

        if len(bboxes)==0: # 对于一些帧中没有object的情况
            # print("no object in the frame")
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
        return object_batch, flow_batch #rgb, flow


    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    # test dataloader
    import torchvision
    from torchvision import datasets
    from torch.utils.data import Dataset, DataLoader
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms

    parser = argparse.ArgumentParser()

    # for flownet2, no need to modify
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    parser.add_argument("--datadir", default='../VAD_datasets/ShanghaiTech/training/frames')
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    # flow 和 yolo 在线计算
    train_data = VadDataset(args, video_folder=args.datadir, bbox_folder=None, dataset="ShanghaiTech", flow_folder=None,
                            device="0", transform=transforms.Compose([transforms.ToTensor()]),
                            resize_height=64, resize_width=64,time_step=16)

    # # 使用保存的.npy
    # train_data = VadDataset(args,video_folder= args.datadir, bbox_folder = "./bboxes/ShanghaiTech/train", flow_folder="./flow/ShanghaiTech/train",
    #                         transform=transforms.Compose([transforms.ToTensor()]),
    #                         resize_height=64, resize_width=64)

    # # 仅在线计算flow
    # train_data = VadDataset(video_folder= datadir, bbox_folder = "./bboxes/ped2/test", flow_folder=None,
    #                         transform=transforms.Compose([transforms.ToTensor()]),
    #                         resize_height=256, resize_width=256, device = device)

    # 仅在线计算yolo
    # train_data = VadDataset(video_folder= datadir, bbox_folder = None,dataset='ped2',
    #                         flow_folder="./flow/ped2/test",
    #                         transform=transforms.Compose([transforms.ToTensor()]),
    #                         resize_height=256, resize_width=256,
    #                         device = device)

    train_loader = DataLoader(dataset=train_data, shuffle=False, batch_size=args.batch_size, collate_fn=my_collate, num_workers=5)

    # 迭代测试：
    for j, (rgb_batch, flow_batch) in enumerate(tqdm(train_loader, desc='train', leave=False)):
        # print(rgb_batch.shape, flow_batch.shape)
        pass 