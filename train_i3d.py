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
from flownet2.models import FlowNet2

import utils
from models.preAE import PreAE, PreAEAttention
from models.unet import UNet
from models.networks import define_G
from models.pix2pix_networks import PixelDiscriminator
# from liteFlownet.lite_flownet import Network, batch_estimate
from losses import *
from vad_dataloader_ped2 import VadDataset
from getFlow import *
from dataset_i3d import *
from models.I3D_STD import *

import torchvision.transforms as transforms
from evaluate_ped2 import *


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
    utils.set_save_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(config['save_path'], 'tensorboard'))
    yaml.dump(config, open(os.path.join(config['save_path'], 'classifier_config.yaml'), 'w'))

    device = torch.device('cuda:' + args.gpu)

    #### make datasets ####
    # train
    norm_dataset = Train_TemAug_Dataset_SHT_I3D(config['dataset_path'], config['train_split'],
                                                config['pseudo_labels'], config['clips_num'],
                                                segment_len=config['segment_len'], type='Normal')

    abnorm_dataset = Train_TemAug_Dataset_SHT_I3D(config['dataset_path'], config['train_split'],
                                                  config['pseudo_labels'], config['clips_num'],
                                                  segment_len=config['segment_len'], type='Abnormal')

    norm_dataloader = DataLoader(norm_dataset, batch_size=config['batch_size'], shuffle=True,drop_last=True, )
    abnorm_dataloader = DataLoader(abnorm_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, )

    test_dataset = Test_Dataset_SHT_I3D(config['dataset_path'], config['test_split'],
                                        config['test_mask_dir'], segment_len=config['segment_len'])
    test_dataloader = DataLoader(test_dataset, batch_size=42, shuffle=False, drop_last=False, )


    #### Model setting ####
    model = I3D_SGA_STD(config['dropout_rate'], config['expand_k'],
                        freeze_backbone=config['freeze_backbone'], freeze_blocks=config['freeze_blocks'],
                        freeze_bn=config['freeze_backbone'],
                        pretrained_backbone=config['pretrained'], pretrained_path=config['pretrained_path'],
                        freeze_bn_statics=True).cuda()
    model = model.train()

    # optimizer setting
    params = list(model.parameters())
    optimizer, lr_scheduler = utils.make_optimizer(
        params, config['optimizer'], config['optimizer_args'])

    lr_policy = lambda epoch: (epoch + 0.5) / (args.warmup_epochs) \
        if epoch < args.warmup_epochs else 1
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_policy)

    criterion = Weighted_BCE_Loss(weights=config['class_reweights'],label_smoothing=config['label_smoothing'], eps=1e-8).cuda()

    # parallel if muti-gpus
    # if torch.cuda.is_available():
    #     model.cuda()
    #
    # if config.get('_parallel'):
    #     model = nn.DataParallel(model)


    # pretrain = False
    #
    # if pretrain:
    #     model.load_state_dict(torch.load('save/ped2_attention_0319mnadall/models/max-frame_auc-model.pth'))
    #     # discriminator.load_state_dict(torch.load('ped2_26000.pth')['net_d'])

    # Training
    utils.log('Start train')
    max_frame_AUC, max_roi_AUC = 0, 0

    save_epoch = 5 if config['save_epoch'] is None else config['save_epoch']
    for epoch in range(config['epochs']):
        if config['freeze_backbone'] and epoch == config['freeze_epochs']:
            model.module.freeze_backbone=False
            model.module.freeze_part_model()
            model.module.freeze_batch_norm()
        if config['freeze_backbone'] and epoch == config['freeze_epochs']:
            model.module.freeze_bn=False
            model.module.freeze_part_model()
            model.module.freeze_batch_norm()

        for step, ((norm_frames, norm_labels), (abnorm_frames, abnorm_labels)) in enumerate(
                zip(norm_dataloader, abnorm_dataloader)):
            # [B,N,C,T,H,W]->[B*N,C,T,W,H]
            frames = torch.cat([norm_frames, abnorm_frames], dim=0).cuda().float()
            frames = frames.view(
                [-1, frames.shape[2], frames.shape[3], frames.shape[4], frames.shape[5]]).cuda().float()
            # labels is with [B,N,2]->[B*N,2]
            labels = torch.cat([norm_labels, abnorm_labels], dim=0).cuda().float()
            labels = labels.view([-1, 2]).cuda().float()

            scores, feat_maps, atten_scores, _ = model(frames)

            scores = scores.view([frames.shape[0], 2])[:, -1]
            atten_scores = atten_scores.view([frames.shape[0], 2])[:, -1]
            labels = labels[:, -1]
            err = criterion(scores, labels)
            atten_err = criterion(atten_scores, labels)

            loss = args.lambda_base * err + args.lambda_atten * atten_err

        lr_scheduler.step()
        utils.log("epoch {}, lr {}".format(epoch, optimizer.param_groups[0]['lr']))
        utils.log('----------------------------------------')

    utils.log('Training is finished')
    utils.log('max_frame_AUC: {}'.format(max_frame_AUC))


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
    utils.set_gpu(args.gpu)
    train(config)
