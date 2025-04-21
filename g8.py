"""
Created on Wed Aug  5 14:09:35 2020

@author: 17505

ResNet一维去噪
输入输出维度无限制
#采用ELU
resnet18
"""

import torch.nn as nn
import torch.nn.functional as F

data_in_channel=6
data_out_channel=10

import torch
import torch.nn as nn


class HybridPooling(nn.Module):
    """自定义平均+最大混合池化层"""

    def __init__(self, pool_size=3, mode='avg_max_sum'):
        super().__init__()
        self.pool_size = pool_size
        self.mode = mode

        # 初始化两种池化方式
        self.avg_pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2)
        self.max_pool = nn.MaxPool2d(pool_size, stride=1, padding=pool_size // 2)

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)

        if self.mode == 'avg_max_sum':
            return 0.5 * (avg_out + max_out)  # 平均加权和
        elif self.mode == 'concat':
            return torch.cat([avg_out, max_out], dim=1)  # 通道维度拼接
        else:
            raise ValueError("Unsupported mode. Choose 'avg_max_sum' or 'concat'")
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ELU(),
            HybridPooling(pool_size=3),  # 插入混合池化
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        # ... 其余部分保持不变 ...
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.elu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(data_in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  3, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 4, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 6, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 3, stride=2)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512*16, 10),
        )

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=3, stride=1, padding=1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        #out = out.squeeze(1)
        #out = out.reshape((out.size(0),data_out_channel,data_size))
        return out


def ResNet34():
    return ResNet(ResidualBlock)

