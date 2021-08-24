import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(torch.nn.Module):
    def __init__(self, t_length = 5, n_channel =3):
        super(Encoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
        
        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )
        
        self.moduleConv1 = Basic(n_channel*(t_length-1), 64)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(256, 512)
        self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)
        
    def forward(self, x):

        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        
        return tensorConv4, tensorConv1, tensorConv2, tensorConv3


class DecoderAttention(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3):
        super(DecoderAttention, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        # self.moduleConv = Basic(2048, 512)
        # # self.moduleConv = Basic(512, 512)
        # self.moduleUpsample4 = Upsample(512, 256)
        #
        # self.moduleDeconv3 = Basic(512, 256)
        # self.moduleUpsample3 = Upsample(256, 128)
        #
        # self.moduleDeconv2 = Basic(256, 128)
        # self.moduleUpsample2 = Upsample(128, 64)
        #
        # self.moduleDeconv1 = Gen(128, n_channel, 64)

        self.moduleConv = Basic(2048, 1024)
        self.moduleUpsample4 = Upsample(1024, 512)

        self.moduleDeconv3 = Basic(512, 256)
        self.moduleUpsample3 = Upsample(256, 128)

        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = Upsample(64, 64)

        self.moduleDeconv1 = Gen(64, n_channel, 64)

    def forward(self, x, skip1, skip2, skip3):
        tensorConv = self.moduleConv(x)
        tensorUpsample4 = self.moduleUpsample4(tensorConv)

        tensorDeconv3 = self.moduleDeconv3(tensorUpsample4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        tensorDeconv2 = self.moduleDeconv2(tensorUpsample3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

        output = self.moduleDeconv1(tensorUpsample2)

        return output

class Decoder(torch.nn.Module):
    def __init__(self, t_length = 5, n_channel =3):
        super(Decoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
                
        
        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )
        
        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels = nc, out_channels=intOutput, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
      
        self.moduleConv = Basic(512, 512)
        # self.moduleConv = Basic(512, 512)
        self.moduleUpsample4 = Upsample(512, 256)

        self.moduleDeconv3 = Basic(512, 256)
        self.moduleUpsample3 = Upsample(256, 128)

        self.moduleDeconv2 = Basic(256, 128)
        self.moduleUpsample2 = Upsample(128, 64)

        self.moduleDeconv1 = Gen(128,n_channel,64)

        # self.moduleConv = Basic(2048, 1024)
        # self.moduleUpsample4 = Upsample(1024, 512)
        #
        # self.moduleDeconv3 = Basic(1024, 512)
        # self.moduleUpsample3 = Upsample(512, 256)
        #
        # self.moduleDeconv2 = Basic(512, 256)
        # self.moduleUpsample2 = Upsample(256, 128)
        #
        # self.moduleDeconv1 = Gen(256, n_channel, 128)

    def forward(self, x, skip1, skip2, skip3):
        tensorConv = self.moduleConv(x)
        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        cat4 = torch.cat((skip3, tensorUpsample4), dim = 1)
        
        tensorDeconv3 = self.moduleDeconv3(cat4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        cat3 = torch.cat((skip2, tensorUpsample3), dim = 1)
        
        tensorDeconv2 = self.moduleDeconv2(cat3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        cat2 = torch.cat((skip1, tensorUpsample2), dim = 1)
        
        output = self.moduleDeconv1(cat2)

        return output
    

class PreAE(torch.nn.Module):
    def __init__(self, n_channel =3,  t_length = 5):
        super(PreAE, self).__init__()

        self.encoder = Encoder(t_length, n_channel)
        self.decoder = Decoder(t_length, n_channel)

    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        fea, skip1, skip2, skip3 = self.encoder(x)
        output = self.decoder(fea, skip1, skip2, skip3)

        return output


class PreAEAttention(torch.nn.Module):
    def __init__(self, n_channel =3,  t_length = 2):
        super(PreAEAttention, self).__init__()

        self.encoder = Encoder(t_length, n_channel)
        self.decoder = DecoderAttention(t_length, n_channel)
        r = 2
        self.attention_meta_learner = nn.Sequential(nn.Linear(1024, int(1024 / r)), nn.ReLU(),
                                                    nn.Linear(int(1024 / r), 1024))  # 这里代表整个网络的frame输入尺寸为256*256

    def get_attention(self, f, b):
        # calc <f,b>
        attention = []
        for i in range(b.shape[0]):
            bi = b[i]
            # cosine_matrix = torch.cosine_similarity(f, bi, dim=0)
            f = f / torch.norm(f, dim=0, p=2)
            bi = bi / torch.norm(bi, dim=0, p=2)
            cosine_matrix = f.t().mm(bi)
            # print("cosine_matrix.shape: ", cosine_matrix.shape)
            p = torch.nn.functional.avg_pool1d(cosine_matrix.unsqueeze(0), kernel_size=cosine_matrix.shape[1],
                                               stride=None)
            # print("p.shape： ", p.shape)
            w = self.attention_meta_learner(p.view(1024))
            # print("w.shape: ", w.shape)
            attention.append(w.view(32, 32))

        attention_map = torch.stack(attention, 0)
        attention_map = torch.mean(attention_map, 0)
        # print("attention_map.shape: ", attention_map.shape)
        return attention_map

    def forward(self, frames, objects):  # frames/objects: N x T x c x h x w

        T = frames.shape[1]
        frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
        objects = objects.reshape(-1, objects.shape[-3], objects.shape[-2], objects.shape[-1])

        f, skip1, skip2, skip3 = self.encoder(frames)  # f: (N * T) x (512) x h' x w'
        ft = f[int(T / 2)].view(f.shape[1], -1)  # c*d1
        # print("f.shape: {}, ft.shape: {}".format(f.shape, ft.shape))

        b, _, _, _ = self.encoder(objects)  # b: (objectnum) x (512) x h'' x w''
        b_view = b.view(b.shape[0], b.shape[1], -1)
        # print("b.shape: {}, bview.shape: {}".format(b.shape, b_view.shape))

        # TODO get_attention
        attention_map = self.get_attention(ft, b_view)
        z = attention_map * f

        # print("z.shape: ", z.shape)

        z = z.unsqueeze(0)
        z = z.reshape(z.shape[0], -1, z.shape[-2], z.shape[-1])
        # out_recurrent, _ = self.recurrent_model(z)
        # out_recurrent = out_recurrent[0].reshape(out_recurrent[0].shape[0], -1, out_recurrent[0].shape[-2],
        #                                          out_recurrent[0].shape[-1])
        # print("out_recurrent.shape: ", out_recurrent.shape)
        out = self.decoder(z, skip1, skip2, skip3)
        return out
