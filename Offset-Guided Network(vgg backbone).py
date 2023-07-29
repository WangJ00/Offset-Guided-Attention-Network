from collections import OrderedDict
import math
from torch.utils import model_zoo
import torch
import torch.nn as nn
import os
import logging
import functools
import random
import numpy as np
import cv2
import torch._utils
import torch.nn.functional as F
from torch.nn import Softmax
from .squeeze_extractor import *

BatchNorm2d=nn.BatchNorm2d
relu_inplace=True
BN_MOMENTUM = 0.1
ALIGN_CORNERS = None
logger = logging.getLogger(__name__)
def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class VGGBackbone(torch.nn.Module):

    def __init__(self, n_classes, pretrained_model: SqueezeExtractor, batch_norm=True):
        super(VGGBackbone, self).__init__()
        last_inp_channels = 64
        lic = 64
        self.softmax = Softmax(dim=3)
        self.copy_feature_info = pretrained_model.get_copy_feature_info()
        self.features = pretrained_model.features

        self.up_layer0 = self._make_up_layer(-1, batch_norm)
        self.up_layer1 = self._make_up_layer(-2, batch_norm)
        self.up_layer2 = self._make_up_layer(-3, batch_norm)
        self.up_layer3 = self._make_up_layer(-4, batch_norm)

        self.up_sampling0 = self._make_up_sampling(-1)
        self.up_sampling1 = self._make_up_sampling(-2)
        self.up_sampling2 = self._make_up_sampling(-3)
        self.up_sampling3 = self._make_up_sampling(-4)

        #find out_channels of the top layer and define classifier
        for f in reversed(self.up_layer3):
            if isinstance(f, nn.Conv2d):
                channels = f.out_channels
                break

        uplayer4 = []
        uplayer4 += [nn.Conv2d(channels, channels, kernel_size=3, padding=1)]
        if batch_norm:
            uplayer4 += [nn.BatchNorm2d(channels)]
        uplayer4 += [nn.ReLU(inplace=True)]
        self.up_layer4 = nn.Sequential(*uplayer4)

        self.up_sampling4 = nn.ConvTranspose2d(channels, channels, kernel_size=4,
                                              stride=2, bias=False)
        self.reducec = nn.Sequential(
            nn.Conv2d(  in_channels = 128,
                        out_channels = last_inp_channels,
                        kernel_size = 1,
                        stride = 1,
                        padding = 0),
            BatchNorm2d(last_inp_channels, momentum = BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace),
            #nn.ConvTranspose2d(lic, lic, 3, stride=2, padding=1),
            nn.Conv2d(  in_channels = lic,
                        out_channels = lic,
                        kernel_size = 3,
                        stride = 4,
                        padding = 1
            ),
            BatchNorm2d(last_inp_channels, momentum = BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace)
        )
        self.ed = nn.Sequential(
            nn.Conv2d(  in_channels = lic,
                        out_channels = lic*2,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1),
            BatchNorm2d(lic*2, momentum=BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace),
            nn.Conv2d(  in_channels = lic*2,
                        out_channels = lic*2,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1),
            BatchNorm2d(lic*2, momentum=BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace),
            nn.Conv2d(  in_channels = lic*2,
                        out_channels = lic*2,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1),
            BatchNorm2d(lic*2, momentum=BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace),
            nn.Conv2d(  in_channels = lic*2,
                        out_channels = lic,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace),
            nn.Conv2d(  in_channels = lic,
                        out_channels = lic,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1)
        )
        self.edgee =nn.Sequential(
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace),
            nn.Conv2d(
                in_channels = lic,
                out_channels = lic,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace),
            nn.Conv2d(
                in_channels = lic,
                out_channels = lic,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace),
            nn.Conv2d(
                in_channels = lic,
                out_channels = lic,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace)
        )
        self.edgediscriminator = nn.Sequential(
            nn.ConvTranspose2d(lic, lic, 3, stride=2, padding=1),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace),
            nn.Conv2d(  in_channels = lic,
                        out_channels = lic,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace),
            nn.Conv2d(
                in_channels = lic,
                out_channels = lic,
                kernel_size = 1,
                stride = 1,
                padding = 0
            ),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace),
            nn.Conv2d(
                in_channels = last_inp_channels,
                out_channels = last_inp_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace),
            nn.Conv2d(
                in_channels = last_inp_channels,
                out_channels = last_inp_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace),
            nn.Conv2d(
                in_channels = last_inp_channels,
                out_channels = last_inp_channels,
                kernel_size = 3,
                stride = 2,
                padding = 1
            )
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(  in_channels = last_inp_channels,
                        out_channels = last_inp_channels,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace),
            nn.Conv2d(
                in_channels = last_inp_channels,
                out_channels = last_inp_channels,
                kernel_size = 1,
                stride = 1,
                padding = 0
            ),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace),
            nn.Conv2d(
                in_channels = last_inp_channels,
                out_channels = last_inp_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            )
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduction = 16
        channel = lic*2
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.fca = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.last_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=lic*2,#in_channels=lic*2,
                out_channels=lic,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=lic,
                out_channels=lic,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=lic,
                out_channels=lic,
                kernel_size=3,
                stride=1,
                padding=1),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=lic,
                out_channels=lic,
                kernel_size=3,
                stride=1,
                padding=1),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=lic,
                out_channels=11,
                kernel_size=1,
                stride=1,
                padding=0)
        )
        self.last_layer2_a = nn.Sequential(
            nn.Conv2d(
                in_channels=lic*2,
                out_channels=lic,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=lic,
                out_channels=lic,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=lic,
                out_channels=lic,
                kernel_size=3,
                stride=1,
                padding=1),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=lic,
                out_channels=lic,
                kernel_size=3,
                stride=1,
                padding=1),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=lic,
                out_channels=11,
                kernel_size=1,
                stride=1,
                padding=0)
        )
        self.convatt = nn.Sequential(
            nn.Conv2d(
                in_channels = lic,
                out_channels = lic,
                kernel_size = 1,
                stride = 1,
                padding = 0
                ),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels = lic,
                out_channels = lic,
                kernel_size = 3,
                stride = 1,
                padding = 1
                ),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=lic,
                out_channels=lic,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        
        self.convatt2 = nn.Sequential(
            nn.Conv2d(
                in_channels = lic*2,
                out_channels = lic,
                kernel_size = 1,
                stride = 1,
                padding = 0
                ),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels = lic,
                out_channels = lic,
                kernel_size = 3,
                stride = 1,
                padding = 1
                ),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=lic,
                out_channels=lic,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        
        self.offout = nn.Sequential(
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=lic,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        self.offsetf = nn.Sequential(
            nn.Conv2d(
                in_channels = lic,
                out_channels = lic,
                kernel_size = 1,
                stride = 1,
                padding = 0
                ),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace),
            nn.Conv2d(
                in_channels = lic,
                out_channels = lic,
                kernel_size = 3,
                stride = 1,
                padding = 1
                ),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace),
            nn.Conv2d(
                in_channels = lic,
                out_channels = lic,
                kernel_size = 3,
                stride = 1,
                padding = 1
                ),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace),
            nn.Conv2d(
                in_channels = lic,
                out_channels = lic,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.Conv2d(
                in_channels=lic,
                out_channels=lic,
                kernel_size=3,
                stride=1,
                padding=1),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=lic,
                out_channels=lic,
                kernel_size=3,
                stride=1,
                padding=1
            )
            
        )
        self.INF = INF
        self.channel_ff = nn.Sequential(
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=lic,
                out_channels=lic,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=lic,
                out_channels=lic,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=lic,
                out_channels=lic,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=lic,
                out_channels=lic,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=lic,
                out_channels=lic,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            BatchNorm2d(lic, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )
        
        #self._initialize_weights()

    def _get_last_out_channels(self, features):
        for idx, m in reversed(list(enumerate(features.modules()))):
            if isinstance(m, nn.Conv2d):
                return m.out_channels
        return 0


    def _make_up_sampling(self, cfi_idx):
        if cfi_idx == -1:
            in_channels = self._get_last_out_channels(self.features)
        else:
            in_channels = self.copy_feature_info[cfi_idx + 1].out_channels

        out_channels = self.copy_feature_info[cfi_idx].out_channels
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,
                                  stride=2, bias=False)

    def _make_up_layer(self, cfi_idx, batch_norm):
        idx = self.copy_feature_info[cfi_idx].index
        for k in reversed(range(0, idx)):
            f = self.features[k]
            channels = self._get_last_out_channels(f)

            if channels == 0:
                continue

            out_channels = self.copy_feature_info[cfi_idx].out_channels
            in_channels = out_channels + channels  # for concatenation.

            layer = []
            layer += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
            if batch_norm:
                layer += [nn.BatchNorm2d(out_channels)]
            layer += [nn.ReLU(inplace=True)]

            return nn.Sequential(*layer)

        assert False
            
    def crossatt(self, xx1, out3):
        x_axis = torch.arange(out3.shape[3]).cuda()
        x_axis = x_axis.repeat(out3.shape[2], 1) #h, w
        x_axis = x_axis.repeat(out3.shape[0], 1, 1) #n, h, w
        y_axis = x_axis.permute(0, 2, 1) #n, w, h
        x_axis = x_axis + out3[:, 1, :, :]
        y_axis = y_axis + out3[:, 0, :, :]
        
        x_axis=torch.unsqueeze(x_axis,1)
        y_axis=torch.unsqueeze(y_axis,1)
        axis = torch.cat((x_axis, y_axis), 1) #n, c, h, w
        m_batchsize, c, height, width = axis.size()
        axis_H = axis.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)#.permute(0, 2, 1) 
        axis_W = axis.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)#.permute(0, 2, 1)
        axis_H = axis_H.repeat(1, 1, height)
        axis_H = torch.reshape(axis_H, (m_batchsize*width, c, height, height))
        axis_W = axis_W.repeat(1, 1, width)
        axis_W = torch.reshape(axis_W, (m_batchsize*height, c, width, width))
        key_H = axis_H.transpose(2, 3)
        key_W = axis_W.transpose(2, 3)
        active_H = torch.abs(axis_H - key_H)
        active_H = (torch.sqrt(torch.pow(active_H[:, 0, :, :],2) + torch.pow(active_H[:, 1, :, :],2) + 0.00001)).view(m_batchsize, width, height, height)
        active_H = active_H*-1
        active_W = torch.abs(axis_W - key_W)
        active_W = (torch.sqrt(torch.pow(active_W[:, 0, :, :],2) + torch.pow(active_W[:, 1, :, :],2) + 0.00001)).view(m_batchsize, height, width, width)
        active_W = active_W*-1
        
        T = 10
        concate = self.softmax(torch.cat([active_H, active_W], 3)/T)
        
        att_H = concate[:,:,:,0:height].contiguous().view(m_batchsize*width,height,height)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)

        xx1_H = xx1.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        xx1_W = xx1.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        out_H = torch.bmm(xx1_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(xx1_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        xx1 = (out_H + out_W)
        return xx1
        
    def load_dict(self): #when loading weight for test, change the load_state_dict path in test.py
        pretrained = 'vgg_the_best.pth' #only for custom pretrain model dict here
        pretrained_dict = torch.load(pretrained)
        model_dict = self.state_dict()
        for k in list(pretrained_dict.keys()):
            pretrained_dict[k.replace("model.", "")] = pretrained_dict.pop(k)
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                   if k in model_dict.keys()}
        for k, _ in pretrained_dict.items():
                    logger.info(
                        '=> loading {} pretrained model {}'.format(k, pretrained))
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        
    def forward(self, x):
        copy_out = []
        o = x
        cpi = self.copy_feature_info[-4:]
        copy_idx = 0

        for i in range(len(self.features)):
            o = self.features[i](o)
            if i == cpi[copy_idx].index - 1:
                copy_out.append(o)
                if copy_idx + 1 < len(cpi):
                    copy_idx += 1
        o = self.up_sampling0(o)
        o = o[:, :, 1:1 + copy_out[3].size()[2], 1:1 + copy_out[3].size()[3]]
        o = torch.cat([o, copy_out[3]], dim=1)
        o = self.up_layer0(o)

        o = self.up_sampling1(o)
        o = o[:, :, 1:1 + copy_out[2].size()[2], 1:1 + copy_out[2].size()[3]]
        o = torch.cat([o, copy_out[2]], dim=1)
        o = self.up_layer1(o)

        o = self.up_sampling2(o)
        o = o[:, :, 1:1 + copy_out[1].size()[2], 1:1 + copy_out[1].size()[3]]
        o = torch.cat([o, copy_out[1]], dim=1)
        o = self.up_layer2(o)

        o = self.up_sampling3(o)
        o = o[:, :, 1:1 + copy_out[0].size()[2], 1:1 + copy_out[0].size()[3]]
        o = torch.cat([o, copy_out[0]], dim=1)
        o = self.up_layer3(o)

        o = self.up_sampling4(o)
        cx = int((o.shape[3] - x.shape[3]) / 2)
        cy = int((o.shape[2] - x.shape[2]) / 2)
        o = o[:, :, cy:cy + x.shape[2], cx:cx + x.shape[3]]
        o = self.up_layer4(o)
        x = self.reducec(o)
        b, c, _, _ = x.size()
        
        out3 = self.offsetf(x)
        out3 = self.offout(out3)
        
        out1a = self.classifier(x)
        out1att = self.crossatt(out1a, out3)
        out1att = self.convatt(out1att)
        out1att = self.crossatt(out1att, out3)
        out1att = self.channel_ff(out1att) + out1att
        
        out1ap = self.avg_pool(out1att)
        outaug = self.ed(x)
        outaugp = self.avg_pool(outaug)
        exitation = torch.cat((outaugp, out1ap), 1)
        y = exitation.view(b, c*2)
        y = self.fc(y).view(b, c*2, 1, 1)
        out1 = torch.cat((outaug * y[:, :c, :, :].expand_as(outaug), out1a * y[:, c:, :, :].expand_as(out1att)), 1)
        
        out2a = self.edgediscriminator(x)+x
        out2a = self.edgee(out2a)+out2a
        out2ap = self.avg_pool(out2a)
        exitation = torch.cat((outaugp, out2ap), 1)
        y = exitation.view(b, c*2)
        y = self.fc2(y).view(b, c*2, 1, 1)
        out2 = torch.cat((outaug * y[:, :c, :, :].expand_as(outaug), out2a * y[:, c:, :, :].expand_as(out2a)), 1)
        
        out1 = self.last_layer1(out1)
        out2 = self.last_layer2_a(out2)
        
        return out1, out2, out3

cfgs = {
    'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 1024, 1024, 'U', 512, 512,
           'U', 256, 256, 'U', 128, 128, 'U', 64, 64]
}

from .vgg import *

def get_seg_model(cfg, **kwargs):
    pretrained=True
    fixed_feature=False
    n_classes=11
    batch_norm = False
    vgg = vgg_16(batch_norm, pretrained, fixed_feature)
    net = VGGBackbone(n_classes, vgg, batch_norm)
    #net.load_dict() #only include this when custom pretrain model is applied, when not included, imagenet pretrained VGG model will be applied for initialization
    return net
