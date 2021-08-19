import cv2
import h5py
import os
from tqdm import tqdm
import numpy as np
from getFlow import FlowNet
import argparse
import torch

"""
For training speed, we translate the video datasets into a single h5py file for reducing the indexing time in Disk
By keeping the compressed type as JPG, we can reduce the memory space
Here, we give the example as translating UCF-Crime training set into a single h5py file, you can modify it for other dataset
OR
You can modify the datasets/dataset.py for directly using the video files for testing!
"""


def Video2FlowH5(h5_path,train_list,flownet,segment_len=16):
    # not multi-thread, may take time
    h=h5py.File(h5_path,'a')
    for path in tqdm(train_list):
        video_frames = os.listdir(path)
        vid_len = len(video_frames)
        for i in tqdm(range(int((vid_len-1)//segment_len))):
            tmp_flow=[]
            key=os.path.split(path)[-1]+'-{0:06d}'.format(i)
            for j in range(segment_len):
                img1 = os.path.join(path,video_frames[i*segment_len+j])
                img2 = os.path.join(path,video_frames[i*segment_len+j+1])
                flow = flownet.get_frame_flow(img1, img2, 256, 256).cpu().numpy().transpose(1,2,0)
                
                bound = 15
                flow = (flow + bound) * (255.0 / (2*bound))
                flow = np.round(flow).astype(int)
                flow[flow >= 255] = 255
                flow[flow <= 0] = 0

                tmp_flow.append(flow)    
            tmp_flow = np.asarray(tmp_flow)
            h.create_dataset(key,data=tmp_flow,chunks=True)
        print(path)

    print('finished!')

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # for flownet2, no need to modify
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    parser.add_argument("--weight", default='flownet2/FlowNet2_checkpoint.pth.tar')
    parser.add_argument("--video_dir", default="../VAD_datasets/ShanghaiTech")
    parser.add_argument("--save_path", default="./data/SHT_Flows.h5")
    # parser.add_argument("--train_txt", default="./data/SH_Train_new.txt")
    parser.add_argument("--gpu", default="")
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    # init flownet
    flownet = FlowNet(args, args.weight, args.gpu)
    video_dir = args.video_dir

    if os.path.exists(args.save_path):
        os.remove(args.save_path)

    train_list = []
    with open("./data/SH_Train_new.txt",'r') as f:
        paths = f.readlines()
        for path in paths:
            path = path.strip('\n')
            vid_name = path.split(',')[0]
            label = path.split(',')[1]
            # print(label)
            # lable 为0说明是正常样本，来自training，为1说明来自testing
            if label == '0':
                vid_path = os.path.join(video_dir, 'training/frames/', vid_name)
            else:
                vid_path = os.path.join(video_dir, 'testing/frames/', vid_name)
            train_list.append(vid_path)

    with open("./data/SH_Test_NEW.txt",'r') as f:
        paths = f.readlines()
        for path in paths:
            path = path.strip('\n')
            vid_name = path.split(',')[0]
            label = path.split(',')[1]
            # print(label)
            # lable 为0说明是正常样本，来自training，为1说明来自testing
            if label == '0':
                vid_path = os.path.join(video_dir, 'training/frames/', vid_name)
            else:
                vid_path = os.path.join(video_dir, 'testing/frames/', vid_name)
            train_list.append(vid_path)             
    
    print(train_list)
    print(len(train_list))

    Video2FlowH5(args.save_path,train_list,flownet,segment_len=16)