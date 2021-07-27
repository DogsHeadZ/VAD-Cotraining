import os
import argparse
import yaml
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import utils
from utils import AverageMeter
from eval_utils import eval, cal_rmse

from dataset_i3d import Train_TemAug_Dataset_SHT_I3D, Test_Dataset_SHT_I3D
from models.I3D_STD import I3D_SGA_STD
from losses import Weighted_BCE_Loss


def eval_epoch(args,model,test_dataloader):
    model = model.eval()
    total_labels, total_scores =  [], []

    for frames,ano_types,idxs,annos in tqdm(test_dataloader):
        frames=frames.float().contiguous().view([-1, 3, frames.shape[-3], frames.shape[-2], frames.shape[-1]]).cuda()

        with torch.no_grad():
            scores, feat_maps = model(frames)[:2]

        if args.ten_crop:
            scores = scores.view([-1, 10, 2]).mean(dim=-2)
        for clip, score, ano_type, idx, anno in zip(frames, scores, ano_types, idxs, annos):
            score = [score.squeeze()[1].detach().cpu().float().item()] * args.segment_len
            anno=anno.detach().numpy().astype(int)
            total_scores.extend(score)
            total_labels.extend(anno.tolist())

    return eval(total_scores,total_labels)

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

    # test
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
    iterator = 0
    save_epoch = 10 if config['save_epoch'] is None else config['save_epoch']
    AUCs,tious,best_epoch,best_tiou_epoch,best_tiou,best_AUC=[],[],0,0,0,0
    for epoch in range(config['epochs']):

        if config['freeze_backbone'] and epoch == config['freeze_epochs']:
            model.module.freeze_backbone=False
            model.module.freeze_bn=False
            model.module.freeze_part_model()
            model.module.freeze_batch_norm()

        Errs, Atten_Errs, Rmses = AverageMeter(), AverageMeter(), AverageMeter()
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
            loss = config['lambda_base'] * err + config['lambda_atten'] * atten_err
            loss.backward()
            if iterator % config['accumulate_step'] == 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                optimizer.step()
                optimizer.zero_grad()

            rmse = cal_rmse(scores.detach().cpu().numpy(), labels.unsqueeze(-1).detach().cpu().numpy())
            Rmses.update(rmse), Errs.update(err), Atten_Errs.update(atten_err)

            iterator += 1
        utils.log('[{}]: err\t{:.4f}\tatten\t{:.4f}'.format(epoch, Errs, Atten_Errs))
        Errs.reset(), Rmses.reset(), Atten_Errs.reset()

        lr_scheduler.step()
        utils.log("epoch {}, lr {}".format(epoch, optimizer.param_groups[0]['lr']))
        utils.log('----------------------------------------')
        if epoch % save_epoch == 0:

            auc=eval_epoch(args,model,test_dataloader)
            AUCs.append(auc)
            if len(AUCs) >= 5:
                mean_auc = sum(AUCs[-5:]) / 5.
                if mean_auc > best_AUC:
                    best_epoch,best_AUC =epoch,mean_auc
                utils.log('best_AUC {} at epoch {}, now {}'.format(best_AUC, best_epoch, mean_auc))

            utils.log('===================')
            if auc > 0.8:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(save_path, 'models/model-epoch-{}-AUC-{}.pth'.format(epoch, auc)))
            model = model.train()

    utils.log('Training is finished')
    utils.log('max_frame_AUC: {}'.format(best_AUC))


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
