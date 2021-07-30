import argparse
import os
import sys
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
from collections import OrderedDict
import glob
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.autograd import Variable
# from flownet2.models import FlowNet

import my_utils
from model.preAE import PreAE
from model.unet import UNet
from losses import *
import torchvision.transforms as transforms

from vad_dataloader_object import VadDataset, my_collate



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def train(config):
    #### set the save and log path ####
    save_path = config['save_path']
    my_utils.set_save_path(save_path)
    my_utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(config['save_path'], 'tensorboard'))
    yaml.dump(config, open(os.path.join(config['save_path'], 'classifier_config.yaml'), 'w'))


    #### make datasets ####
    # train
    train_folder = config['dataset_path'] + config['train_dataset_type'] + "/training/frames"
    test_folder = config['dataset_path'] + config['train_dataset_type'] + "/testing/frames"

    # Loading dataset
    train_dataset_args = config['train_dataset_args']
    test_dataset_args = config['test_dataset_args']

    train_dataset = VadDataset(args, video_folder=train_folder, bbox_folder=None, dataset="ShanghaiTech", flow_folder=None,
                            device="0", transform=transforms.Compose([transforms.ToTensor()]),
                            resize_height=64, resize_width=64, time_step=train_dataset_args['t_length'])

    # test_dataset = VadDataset(args, video_folder=test_folder, bbox_folder=None, dataset="ShanghaiTech", flow_folder=None,
    #                         device="0", transform=transforms.Compose([transforms.ToTensor()]),
    #                         resize_height=64, resize_width=64, time_step=1, num_pred=0)


    train_dataloader = DataLoader(train_dataset, batch_size=train_dataset_args['batch_size'],
                                  shuffle=True, num_workers=train_dataset_args['num_workers'], drop_last=True, collate_fn=my_collate)
    # test_dataloader = DataLoader(test_dataset, batch_size=test_dataset_args['batch_size'],
    #                              shuffle=False, num_workers=test_dataset_args['num_workers'], drop_last=False)

    # Model setting
    rgb_model =PreAE(train_dataset_args['c'], train_dataset_args['t_length']+1)
    flow_model = PreAE(train_dataset_args['c'], train_dataset_args['t_length']+1)


    # optimizer setting
    params = list(rgb_model.parameters())
    rgb_optimizer_G, rgb_lr_scheduler = my_utils.make_optimizer(
        params, config['optimizer'], config['optimizer_args'])

    params = list(flow_model.parameters())
    flow_optimizer_G, flow_lr_scheduler = my_utils.make_optimizer(
        params, config['optimizer'], config['optimizer_args'])

    # set loss, different range with the source version, should change
    lam_int = float(config['lam_int'])
    lam_gd = float(config['lam_gd'])
    # lam_op = float(config['lam_op'])
    alpha = 1
    l_num = 2
    gd_loss = Gradient_Loss(alpha, train_dataset_args['c'])
    int_loss = Intensity_Loss(l_num)
    # op_loss = Flow_Loss()

    # parallel if muti-gpus
    if torch.cuda.is_available():
        rgb_model.cuda()
        flow_model.cuda()
    if config.get('_parallel'):
        rgb_model = nn.DataParallel(rgb_model)
        flow_model = nn.DataParallel(flow_model)
        

    # Training
    my_utils.log('Start train')
    max_frame_AUC, max_roi_AUC = 0,0
    # base_channel_num  = train_dataset_args['c'] * (train_dataset_args['t_length'] - 1)
    save_epoch = 5 if config['save_epoch'] is None else config['save_epoch']
    for epoch in range(config['epochs']):
        rgb_model.train()
        flow_model.train()
        for j, (rgb,flow) in enumerate(tqdm(train_dataloader, desc='train', leave=False)):
            # print(rgb.shape, flow.shape)
            rgb = rgb.cuda()
            flow = flow.cuda()
            rgb_input = rgb.contiguous().view(-1, rgb.shape[2], rgb.shape[3], rgb.shape[4], rgb.shape[5])
            flow_input = flow.contiguous().view(-1, rgb.shape[2], flow.shape[3], flow.shape[4], flow.shape[5])

            print(rgb_input.shape)

            rgb_target = rgb[:,:,-1]
            flow_target = flow[:,:,-1]
            # print(rgb_target.shape)
            rgb_target = rgb_target.contiguous().view(-1, rgb_target.shape[-3], rgb_target.shape[-2], rgb_target.shape[-1])
            flow_target = flow_target.contiguous().view(-1, flow_target.shape[-3], flow_target.shape[-2], flow_target.shape[-1])

            rgb_outputs = rgb_model(rgb_input)
            flow_outputs = flow_model(flow_input)
            
            g_int_loss = int_loss(rgb_outputs, rgb_target)
            g_gd_loss = gd_loss(rgb_outputs, rgb_target)
            rgb_g_loss = g_int_loss + g_gd_loss

            flow_g_loss = int_loss(flow_outputs, flow_target)

            rgb_optimizer_G.zero_grad()
            g_loss.backward()

            flow_optimizer_G.zero_grad()
            flow_g_loss.backward()

            rgb_optimizer_G.step()
            flow_optimizer_G.step()

            # train_psnr = my_utils.psnr_error(outputs,target)

        if rgb_lr_scheduler is not None:
            rgb_lr_scheduler.step()
        if flow_lr_scheduler is not None:
            flow_lr_scheduler.step()
        # TODO: val



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    # for flownet2, no need to modify
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    if args.tag is not None:
        config['save_path'] += ('_' + args.tag)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu
    else:
        torch.cuda.set_device(int(args.gpu))
    my_utils.set_gpu(args.gpu)
    train(config)
