import argparse
import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import cv2


from flownet2.models import FlowNet2  # the path is depended on where you create this module
from flownet2.utils_flownet2.frame_utils import read_gen  # the path is depended on where you create this module
from flownet2.utils_flownet2 import flow_utils, tools
import torchvision.transforms as transforms

class FlowNet:
    def __init__(self, args, weight, device):   
        """init"""
        if device == "":
            self.device = "cpu"
        else:     
            self.device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu") 

        self.net = FlowNet2(args).to(self.device)
        # load the state_dict
        dict_ = torch.load(weight)
        self.net.load_state_dict(dict_["state_dict"])  

    @staticmethod
    def writeFlow(name, flow):
        # np.save(name, flow)
        torch.save(flow, name)

    @staticmethod
    def readFlow(name):
        flow = np.load(name, allow_pickle=True)
        return flow


    def get_frame_flow(self, img1, img2, width, height):
        pim1 = cv2.imread(img1)
        pim2 = cv2.imread(img2)
        pim1 = cv2.resize(pim1, (width, height) )
        pim2 = cv2.resize(pim2, (width, height) )
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        pim1 = transform(pim1).unsqueeze(dim=0).to(self.device)
        pim2 = transform(pim2).unsqueeze(dim=0).to(self.device)

        pred_flow_esti_tensor = torch.cat([pim1.view(-1, 3, 1, pim1.shape[-2], pim1.shape[-1]),
                                        pim2.view(-1, 3, 1, pim2.shape[-2], pim2.shape[-1])], 2)

        result = self.net(pred_flow_esti_tensor * 255.0).squeeze()
        # data = result.data.cpu().numpy().transpose(1, 2, 0)


        return result.data



if __name__ == '__main__':
    # obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    parser.add_argument("--weight", default='flownet2/FlowNet2_checkpoint.pth.tar')
    parser.add_argument("--datadir", default="../VAD_datasets/ShanghaiTech/training/frames")
    parser.add_argument("--save_path", default="./flow/ShanghaiTech/train/")
    parser.add_argument("--gpu", default="")
    args = parser.parse_args()
    # print(args)

    # initial a Net
    flownet = FlowNet(args, args.weight, args.gpu)

    # 计算所有图像的flow
    videos = args.datadir
    save_path = args.save_path
    video_list = sorted(os.listdir(videos))
    for video in video_list:
        frame_list = sorted(os.listdir(os.path.join(videos, video)))
        for i in range( 0, len(frame_list)-1 ):
            frame1 = os.path.join(videos, video, frame_list[i])
            frame2 = os.path.join(videos, video, frame_list[i+1])
            # print(frame2)
            flow = flownet.get_frame_flow( frame1, frame2, 512, 384 ) 
            # print(flow.shape) shape:[2,H,W]

            dir_ = os.path.join(save_path, video)
            if not os.path.exists(dir_):
                os.mkdir(dir_)
            flownet.writeFlow(dir_ + "/" + frame_list[i].split('.')[0] + ".pt", flow.cpu() )

        print("save: ", video)

    
             


    
