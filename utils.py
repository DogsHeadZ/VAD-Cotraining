import os
import shutil
import time
import math

from sklearn.metrics import roc_auc_score

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

_log_path = None

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.vals=[]

    def __format__(self, format_spec):
        f=0
        if len(self.vals)!=0:
            f=(sum(self.vals)/len(self.vals))
        return ('{:'+format_spec+'}').format(f)

    def val(self):
        if len(self.vals) != 0:
            f = sum(self.vals) / len(self.vals)
        else:
            f=0
        return f

    def update(self,val):
        if isinstance(val,np.ndarray):
            self.vals.append(val[0])
        elif isinstance(val,np.float64):
            self.vals.append(val)
        else:
            self.vals.append(val.detach().cpu().item())


def psnr(mse):

    return 10 * math.log10(1 / mse)


def point_score(outputs, imgs):
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0] + 1) / 2, (imgs[0] + 1) / 2)
    normal = (1 - torch.exp(-error))
    score = (torch.sum(normal * loss_func_mse((outputs[0] + 1) / 2, (imgs[0] + 1) / 2)) / torch.sum(normal)).item()
    return score


def point_score(outputs, imgs):
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0] + 1) / 2, (imgs[0] + 1) / 2)
    normal = (1 - torch.exp(-error))
    score = (torch.sum(normal * loss_func_mse((outputs[0] + 1) / 2, (imgs[0] + 1) / 2)) / torch.sum(normal)).item()
    return score


def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr - min_psnr))


def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr - min_psnr)))


def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))

    return anomaly_score_list


def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))

    return anomaly_score_list


def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc


def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha * list1[i] + (1 - alpha) * list2[i]))

    return list_result


def log10(t):
    """
    Calculates the base-10 log of each element in t.
    @param t: The tensor from which to calculate the base-10 log.
    @return: A tensor with the base-10 log of each element in t.
    """

    numerator = torch.log(t)
    denominator = torch.log(torch.FloatTensor([10.])).cuda()
    return numerator / denominator


def psnr_error(gen_frames, gt_frames):
    """
    Computes the Peak Signal to Noise Ratio error between the generated images and the ground
    truth images.
    @param gen_frames: A tensor of shape [batch_size, height, width, 3]. The frames generated by the
                       generator model.
    @param gt_frames: A tensor of shape [batch_size, height, width, 3]. The ground-truth frames for
                      each frame in gen_frames.
    @return: A scalar tensor. The mean Peak Signal to Noise Ratio error over each frame in the
             batch.
    """
    shape = list(gen_frames.shape)
    num_pixels = (shape[1] * shape[2] * shape[3])
    gt_frames = (gt_frames + 1.0) / 2.0
    gen_frames = (gen_frames + 1.0) / 2.0
    square_diff = (gt_frames - gen_frames)**2

    batch_errors = 10 * log10(1. / ((1. / num_pixels) * torch.sum(square_diff, [1, 2, 3])))
    return torch.mean(batch_errors)

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t >= 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def visualize_dataset(dataset, name, writer, n_samples=16):
    demo = []
    for i in np.random.choice(len(dataset), n_samples):
        demo.append(dataset[i][0])
    writer.add_images('visualize_' + name, torch.stack(demo))
    writer.flush()


def set_gpu(gpu):
    print('set gpu:', gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def set_save_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def make_optimizer(params, name, args):

    if name == 'SGD':
        optimizer = optim.SGD(
            params,
            lr=args['lr'],
            momentum=args['mom'],
            weight_decay=args['weight_decay']
        )
    else:
        optimizer = optim.Adam(params, args['lr'], weight_decay=args['weight_decay'], betas=args['betas'])

    if args['lr_scheduler'] == 'StepLR':

        lr_scheduler = optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=int(args['step_size']),
                            gamma=args['gamma']
                        )

    elif args['lr_scheduler'] == 'MultiStepLR':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
                            optimizer,
                            milestones=args['step_size'],
                            gamma=args['gamma'],
                        )

    elif args['lr_scheduler'] == 'CosineAnnealingLR':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['T_max'])

    else:
        lr_scheduler = None
    return optimizer, lr_scheduler


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def random_perturb(v_len, num_segments):
    """
    Given the length of video and sampling number, which segments should I choose?
    Random sampling is used.
    :param v_len: length of video
    :param num_segments: expected number of segments
    :return: a list of indices to sample
    """
    random_p = np.arange(num_segments) * v_len / num_segments
    for i in range(num_segments):
        if i < num_segments - 1:
            if int(random_p[i]) != int(random_p[i + 1]):
                random_p[i] = np.random.choice(range(int(random_p[i]), int(random_p[i + 1]) + 1))
            else:
                random_p[i] = int(random_p[i])
        else:
            if int(random_p[i]) < v_len - 1:
                random_p[i] = np.random.choice(range(int(random_p[i]), v_len))
            else:
                random_p[i] = int(random_p[i])
    return random_p.astype(int)