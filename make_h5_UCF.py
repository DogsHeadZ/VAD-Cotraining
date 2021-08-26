'''
Author: lyx
Date: 2021-08-24 17:29:38
LastEditTime: 2021-08-25 19:11:22
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /VAD-Cotraing/make_h5/make_h5_UCF.py
'''
  
import cv2
import h5py
import os
from tqdm import tqdm
import numpy as np

"""
For training speed, we translate the video datasets into a single h5py file for reducing the indexing time in Disk
By keeping the compressed type as JPG, we can reduce the memory space
Here, we give the example as translating UCF-Crime training set into a single h5py file, you can modify it for other dataset
OR
You can modify the datasets/dataset.py for directly using the video files for testing!
"""

def Video2ImgH5(video_dir,h5_path,train_list,segment_len=16,max_vid_len=2000):
    # not multi-thread, may take time
    h=h5py.File(h5_path,'a')
    for path in tqdm(train_list):
        # print(path)
        vc=cv2.VideoCapture(path)
        vid_len=vc.get(cv2.CAP_PROP_FRAME_COUNT)
        print(path)
        if vid_len == 0:
            print("vid_len = 0! read video error!")
            exit(1)
        for i in range(int(vid_len//segment_len)):
            tmp_frames=[]
            key=path.split('/')[-1].split('.')[0]+'-{0:06d}'.format(i)
            for j in range(segment_len):
                ret,frame=vc.read()
                _,frame=cv2.imencode('.JPEG',frame)
                frame=np.array(frame).tostring()
                if ret:
                    tmp_frames.append(frame)
                else:
                    print('Bug Reported!')
                    exit(-1)
            tmp_frames = np.asarray(tmp_frames)
            h.create_dataset(key,data=tmp_frames,chunks=True)
        

    print('finished!')

if __name__=='__main__':
    video_dir='/data0/JY/zwh/AllDatasets/UCF/videos/'
    h5_file_path='../datasets/UCF-Frames.h5'
    txt_path='/data0/JY/zwh/AllDatasets/UCF/UCF_Crimes-Train-Test-Split/Anomaly_Detection_splits/Anomaly_Train.txt'
    
    train_list=[]
    with open(txt_path,'r')as f:
        paths=f.readlines()
        for path in paths:
            # ano_type=path.strip().split('/')[0]
            # if 'Normal' in ano_type:
                # path='Normal/'+path.strip().split('/')[-1]
                # continue
            train_list.append(video_dir+path.strip())
    print(train_list)
    Video2ImgH5(video_dir,h5_file_path,train_list,segment_len=16)
    
    test_txt = "/data0/JY/zwh/AllDatasets/UCF/UCF_Crimes-Train-Test-Split/Anomaly_Detection_splits/Anomaly_Test.txt"
    test_list = []
    with open(test_txt,'r')as f:
        paths=f.readlines()
        for path in paths:
            ano_type=path.strip().split('/')[0]
            # if 'Normal' in ano_type:
                # path='Normal/'+path.strip().split('/')[-1]
                # continue
            test_list.append(video_dir+path.strip())
    print(test_list)
    Video2ImgH5(video_dir,h5_file_path,test_list,segment_len=16)