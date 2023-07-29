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


BatchNorm2d=nn.BatchNorm2d
relu_inplace=True
BN_MOMENTUM = 0.1
ALIGN_CORNERS = None
logger = logging.getLogger(__name__)
def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

def load_weights_sequential(target, source_state):
    new_dict = OrderedDict()
    for (k1, v1), (k2, v2) in zip(target.state_dict().items(), source_state.items()):
        new_dict[k1] = v2
    target.load_state_dict(new_dict)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.softmax = Softmax(dim=3)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        lic = 64
        last_inp_channels = lic
        self.reducec = nn.Sequential(
            nn.Conv2d(  in_channels = 2048,
                        out_channels = last_inp_channels,
                        kernel_size = 1,
                        stride = 1,
                        padding = 0),
            BatchNorm2d(last_inp_channels, momentum = BN_MOMENTUM),
            nn.ReLU(inplace = relu_inplace),
            nn.ConvTranspose2d(lic, lic, 3, stride=2, padding=1),
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
        self.last_layer2 = nn.Sequential(
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    
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
        axis_H = axis.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        axis_W = axis.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
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
    
    def ccatt(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = x
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = x
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = x
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        return (out_H + out_W)
        
    def disatt(self, xx1, out3):
        x_axis = torch.arange(out3.shape[3]).cuda()
        x_axis = x_axis.repeat(out3.shape[2], 1) #h, w
        x_axis = x_axis.repeat(out3.shape[0], 1, 1) #n, h, w
        y_axis = x_axis.permute(0, 2, 1) #n, w, h
        
        x_axis=torch.unsqueeze(x_axis,1)
        y_axis=torch.unsqueeze(y_axis,1)
        axis = torch.cat((x_axis, y_axis), 1) #n, c, h, w
        m_batchsize, c, height, width = axis.size()
        axis_H = axis.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        axis_W = axis.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
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
    
    def concatoff(self, out1att, out3):
        out3a = out3
        x_axis = torch.arange(out3a.shape[3]).cuda()
        x_axis = x_axis.repeat(out3a.shape[2], 1) #h, w
        x_axis = x_axis.repeat(out3a.shape[0], 1, 1) #n, h, w
        y_axis = x_axis.permute(0, 2, 1) #n, w, h
        x_axis = x_axis + out3a[:, 1, :, :]
        y_axis = y_axis + out3a[:, 0, :, :]
        
        x_axis=torch.unsqueeze(x_axis,1)
        y_axis=torch.unsqueeze(y_axis,1)
        axis = torch.cat((x_axis, y_axis), 1)
        out1att = torch.cat((out1att, axis/10), 1)
        return out1att
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)
        x = self.layer4(x_3)
        x = self.reducec(x)
        b, c, _, _ = x.size()
        
        out3 = self.offsetf(x)
        out3 = self.offout(out3)
        
        out1a = self.classifier(x)
        out1att = self.crossatt(out1a, out3)
        out1att = self.convatt(out1att)
        out1att = self.crossatt(out1att, out3)
        out1att = self.channel_ff(out1att) + out1att
        out1a = out1att
        
        out1ap = self.avg_pool(out1a)
        outaug = self.ed(x)
        outaugp = self.avg_pool(outaug)
        exitation = torch.cat((outaugp, out1ap), 1)
        y = exitation.view(b, c*2)
        y = self.fc(y).view(b, c*2, 1, 1)
        out1 = torch.cat((outaug * y[:, :c, :, :].expand_as(outaug), out1a * y[:, c:, :, :].expand_as(out1a)), 1)
        
        
        out2a = self.edgediscriminator(x)+x
        out2a = self.edgee(out2a)+out2a
        out2ap = self.avg_pool(out2a)
        exitation = torch.cat((outaugp, out2ap), 1)
        y = exitation.view(b, c*2)
        y = self.fc2(y).view(b, c*2, 1, 1)
        out2 = torch.cat((outaug * y[:, :c, :, :].expand_as(outaug), out2a * y[:, c:, :, :].expand_as(out2a)), 1)

        out1 = self.last_layer1(out1)
        out2 = self.last_layer2(out2)
        return out1, out2, out3

    def init_weights(self):
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            pretrained = 'respre.pth'#'the_best.pth'#'respre.pth'
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

def resnet18(pretrained=True):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=True):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model


def resnet101(pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    return model


def resnet152(pretrained=True):
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet152']))
    return model

def get_seg_model(cfg, **kwargs):
    net = resnet101()
    net.init_weights()
    return net
