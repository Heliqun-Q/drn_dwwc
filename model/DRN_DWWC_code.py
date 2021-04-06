# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:28:56 2021

@author: 17845
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class FirstBlock(nn.Module):

    def __init__(self, in_planes, planes, reweight_size=64, stride=2):
        super(FirstBlock, self).__init__()
        #卷积分支
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        #动态调整分支
        self.reweight = torch.nn.Parameter(torch.randn((1, 1, reweight_size, 1)))

    def forward(self, x):
        out = self.conv1(x)
        out += F.avg_pool2d(torch.mul(x, self.reweight), 2)
        return out
    
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, reweight_size=4, stride=1):
        super(BasicBlock, self).__init__() 
        #卷积分支
        #self.stride = 1
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        #残差分支
        self.shortcut = nn.Sequential()
        #动态调整分支
        self.reweight = torch.nn.Parameter(torch.randn((1, 1, reweight_size, 1)))
        self.dynamic = nn.Sequential()

    def forward(self, x):
        dy = F.relu(self.bn1(x))
        out = self.conv1(dy)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        out += self.dynamic(torch.mul(dy, self.reweight))

        return out

class AdjustBlock(nn.Module):
    def __init__(self, in_planes, planes, reweight_size=4, stride=2):

        super(AdjustBlock, self).__init__() 
        #卷积分支
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
 
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        #残差分支
        self.shortcut = nn.Sequential()
        #动态调整分支
        self.reweight = torch.nn.Parameter(torch.randn((1,1,reweight_size,1)))
        self.dynamic = nn.Sequential()
            
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
                )
    def forward(self, x):
        dy = F.relu(self.bn1(x))
        out = self.conv1(dy)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        dyna = self.dynamic(torch.mul(dy, self.reweight))
        if self.stride != 1:
            dyna1 = F.avg_pool2d(dyna, self.stride)
            if self.in_planes != self.planes:
                # 补零填充
                dyna2 = F.pad(dyna1, pad=(0,0,0,0,self.in_planes,0), mode="constant",value=0)
                out += dyna2
            if self.in_planes == self.planes:
                out += dyna1
            
        else:
            out += dyna

        return out

class DRN_DWWC(nn.Module):

    def __init__(self, num_classes=10, m=2):   
        super(DRN_DWWC, self).__init__()
        self.layer1 = FirstBlock(in_planes=1, planes=m)
        self.layer2 = AdjustBlock(in_planes=m, planes=m, reweight_size=32, stride=2)
        self.layer3 = BasicBlock(in_planes=m, planes=m, reweight_size=16, stride=1)
        self.layer4 = BasicBlock(in_planes=m, planes=m, reweight_size=16, stride=1)
        self.layer5 = AdjustBlock(in_planes=m, planes=2*m, reweight_size=16, stride=2)
        self.layer6 = BasicBlock(in_planes=2*m, planes=2*m, reweight_size=8, stride=1)
        self.layer7 = BasicBlock(in_planes=2*m, planes=2*m, reweight_size=8, stride=1)
        self.layer8 = AdjustBlock(in_planes=2*m, planes=4*m, reweight_size=8, stride=2)
        self.layer9 = BasicBlock(in_planes=4*m, planes=4*m, reweight_size=4, stride=1)
        self.layer10 = BasicBlock(in_planes=4*m, planes=4*m, reweight_size=4, stride=1)
        self.bn = nn.BatchNorm2d(4*m)
        
        self.linear = nn.Linear(4*m*1*1, num_classes)

    # def _make_layer(self, block, planes, stride):
    #     layers = []
    #     layers.append(block(self.in_planes, planes, stride))
    #     #self.in_planes = planes
    #     return nn.Sequential(*layers)  #如果号加在了是实参上，代表的是将输入迭代器拆成一个个元素
    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        #out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

