# -*- coding: utf-8 -*-
"""
discriminator of GAN
Create on Wednesday August 8

@author:loop
"""
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, K=10, T=10):
        super(Discriminator, self).__init__()

        self.K = K
        self.T = T

        # define discriminator of convolution
        #self.conv1 = nn.Conv2d(K + T, 64, kernel_size=5, padding=2, stride=2)  # ((128-5+2*2)+1)/2=64
        self.conv1 = nn.Conv2d(K, 64, kernel_size=5, padding=2, stride=2)  # ((128-5+2*2)+1)/2=64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2)  # ((64-5+2*2)+1)/2=32
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2)  # ((32-5+2*2)+1)/2=16
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=2)  # ((16-5+2*2)+1)/2=8
        self.fc = nn.Linear(8 * 8 * 512, 1)

    def forward(self, x):
        """
        compute discriminator
        :param x: input data of [batch, channel, H, W]
        :return: discriminate result and logits (W*X matrix, not need sigmoid)
        """

        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)

        x = x.view(-1, 8 * 8 * 512)
        x = self.fc(x)

        return F.sigmoid(x)
