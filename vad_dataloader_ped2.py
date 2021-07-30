import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch
import torch.utils.data as data
from getROI import *
from getFlow import *
import torch.nn.functional as F


def np_load_frame_roi(filename, resize_height, resize_width, bbox):
    (xmin, ymin, xmax, ymax) = bbox
    image_decoded = cv2.imread(filename)
    image_decoded = image_decoded[ymin:ymax, xmin:xmax]

    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    # image_resized = image_resized.astype(dtype=np.float32)
    # image_resized = (image_resized / 127.5) - 1.0
    return image_resized


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
    image_resized = image_resized.astype(dtype=np.float32)
    # image_resized = (image_resized )/255.0

    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


class VadDataset(data.Dataset):
    def __init__(self, flowargs, video_folder, transform, resize_height, resize_width, dataset='', time_step=4, num_pred=1,
                 bbox_folder=None, device='cuda:0', flow_folder=None):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.dataset = dataset  # ped2 or avenue or ShanghaiTech

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

    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            # video_name = video.split('/')[-1]
            video_name = os.path.split(video)[-1]
            for i in range(len(self.videos[video_name]['frame']) - self._time_step):  # 减掉_time_step为了刚好能够滑窗到视频尾部
                frames.append(self.videos[video_name]['frame'][i])  # frames存储着训练时每段视频片段的首帧，根据首帧向后进行滑窗即可得到这段视频片段

        return frames

    def __getitem__(self, index):
        # video_name = self.samples[index].split('/')[-2]      #self.samples[index]取到本次迭代取到的视频首帧，根据首帧能够得到其所属类别及图片名
        # frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])
        # windows

        video_name = os.path.split(os.path.split(self.samples[index])[0])[1]
        frame_name = int(os.path.split(self.samples[index])[1].split('.')[-2])


        frame_batch = []
        for i in range(self._time_step + self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name + i], self._resize_height,
                                  self._resize_width)  # 根据首帧图片名便可加载一段视频片段
            if self.transform is not None:
                frame_batch.append(self.transform(image))
        frame_batch = torch.stack(frame_batch, dim=0) # 大小为[_time_step+num_pred, c, _resize_height, _resize_width]

        return frame_batch

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
    args = parser.parse_args()

    batch_size = 1
    datadir = "../AllDatasets/ped2/testing/frames"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # flow 和 yolo 在线计算
    # train_data = VadDataset(args, video_folder=datadir, bbox_folder=None, dataset="avenue", flow_folder=None,
    #                         device=device, transform=transforms.Compose([transforms.ToTensor()]),
    #                         resize_height=256, resize_width=256)

    # 使用保存的.npy
    train_data = VadDataset(args,video_folder= datadir, bbox_folder = "./bboxes/ped2/test", flow_folder="./flow/ped2/test",
                            transform=transforms.Compose([transforms.ToTensor()]),
                            resize_height=256, resize_width=256)

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

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    # for j, (imgs, bbox, flow) in enumerate(train_loader):
    #     print(j)

    unloader = transforms.ToPILImage()

    frames, objects, bboxes, flow= next(iter(train_loader))

    print(frames.shape, objects.shape, flow.shape)
    print(bboxes[0])
    objects = objects.squeeze(0)
    # 显示一个batch, pil显示的颜色是有问题的，只是大概看一下
    index = 1
    for i in range(objects.shape[0]):
        for j in range(1):
            plt.subplot(objects.shape[0], 1, index)
            index += 1
            img = objects[i,j,:,:,:].mul(255).byte()
            img = img.cpu().numpy().transpose((1,2,0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = X[i,j,:,:,:].cpu().clone()
            # img = unloader(img)
            # img = np.array(img)
            # for bbox in bboxes:
            #     plot_one_box(bbox, img)
            plt.imshow(img)

    plt.savefig('objects.jpg')
    plt.close()

    # index = 1
    # for i in range(batch2.shape[0]):
    #     for j in range(1):
    #         plt.subplot(batch2.shape[0], 1, index)
    #         index += 1
    #         img = batch2[i,j,:,:,:].mul(255).byte()
    #         img = img.cpu().numpy().transpose((1,2,0))
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         # img = X[i,j,:,:,:].cpu().clone()
    #         # img = unloader(img)
    #         # img = np.array(img)
    #         for bbox in bboxes:
    #              plot_one_box(bbox, img)
    #         plt.imshow(img)
    #
    # print(index)
    # plt.savefig('objectloss2.jpg')

    for i in range(5):
        img = cv2.imread(os.path.join(datadir, "01", str(i).zfill(3))+".jpg")
        img = cv2.resize(img,(256,256))
        draw_bbox(img, bboxes, (255,0,0), 2)
        cv2.imwrite("globalmemory{}.jpg".format(i), img)
