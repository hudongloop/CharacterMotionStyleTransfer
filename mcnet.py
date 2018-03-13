# -*- coding: utf-8 -*-
"""
building mcnet
Create on Tues August 1

@author: loop
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

# Net structure
class MCnet(nn.Module):
    def __init__(self, K=10, T=10):
        super(MCnet, self).__init__()

        self.K = K
        self.T = T

        # ConvLSTM
        self.cell = nn.ConvLSTMCell(256, 256, kernel_size=3)

        # motion encoder
        self.convM_1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)    # (128-5+2*2)+1=128
        self.poolM_1 = nn.MaxPool2d(2, 2)                            # 128/2=64
        self.convM_2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)  # (64-5+2*2)+1=64
        self.poolM_2 = nn.MaxPool2d(2, 2)                            # 64/2=32
        self.convM_3 = nn.Conv2d(128, 256, kernel_size=7, padding=3) # (32-7+3*2)+1=32
        self.poolM_3 = nn.MaxPool2d(2, 2)                            # 32/2=16

        # content encoder
        self.convC1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # (128-3+1*2)+1=128
        self.convC1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.poolC_1 = nn.MaxPool2d(2, 2, return_indices=True)      # 128/2=64

        self.convC2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # (64-3+1*2)+1=64
        self.convC2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.poolC_2 = nn.MaxPool2d(2, 2, return_indices=True)       # 64/2=32

        self.convC3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # (32-3+1*2)+1=64
        self.convC3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.convC3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.poolC_3 = nn.MaxPool2d(2, 2, return_indices=True)        # 64/2=16

        # combination laters
        self.convComb_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1) # (32-3+1*2)+1=32  256+256=512
        self.convComb_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1) # (32-3+1*2)+1=32
        self.convComb_3 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # (32-3+1*2)+1=32

        # residual (3 coordinate convolution produce residual of each layer) all size not change
        self.convRes1_1 = nn.Conv2d(64+64, 64, kernel_size=3, padding=1)
        self.convRes1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.convRes2_1 = nn.Conv2d(128+128, 128, kernel_size=3, padding=1)
        self.convRes2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.convRes3_1 = nn.Conv2d(256+256, 256, kernel_size=3, padding=1)
        self.convRes3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # decoder (operation is contrary to content encoder)
        self.depool3 = nn.MaxUnpool2d(2, 2)
        self.deconvC3_3 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        self.deconvC3_2 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        self.deconvC3_1 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)

        self.depool2 = nn.MaxUnpool2d(2, 2)
        self.deconvC2_2 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.deconvC2_1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)

        self.depool1 = nn.MaxUnpool2d(2, 2)
        self.deconvC1_2 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.deconvC1_1 = nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1)

    def forward(self, diff_in, xt):

        # motion encoder
        for t in xrange(self.K - 1):
            enc_h, res_m, gram_s = self.motion_enc(diff_in[:, :, :, :, t])
            h_dyn, state = self.cell(enc_h)

        # content encoder, decoder and video generation
        pred = [] # save prediction
        gram = [] # save gram matrix to compute gram_s,gram_c,gram_f
        for t in xrange(self.T):
            h_cont, res_c, indices_c, gram_c = self.content_enc(xt)
            h_comb = self.comb_layers(h_dyn, h_cont)
            res_connect = self.residual(res_m, res_c)
            x_cont, gram_f = self.content_dec(h_comb, res_connect, indices_c)

            # add sequence channel
            pred.append(x_cont.view(x_cont.size(0), x_cont.size(1),
                                    x_cont.size(2), x_cont.size(3), 1))
            # add gram matrix to compute
            gram.append((gram_s, gram_c, gram_f))

            # convert gray image
            x_gray = x_cont.add(1.0).div(2.0)
            xt_gray = xt.add(1.0).div(2.0)

            #compute subtraction of next frame and update next xt input
            diff_next_in = x_gray - xt_gray
            xt = x_cont

            # update feature of motion encoder with ConvLSTM
            enc_h, res_m, gram_s = self.motion_enc(diff_next_in)
            h_dyn, state = self.cell(enc_h)

        return pred, gram


    def motion_enc(self, x):
        """
        :param x: input data of subtraction between t and t-1
        :return: result data of convolution and residual data
        """
        res_in = []
        gram_in = [] #Gram matrix for motion encoder

        x = F.relu(self.convM_1(x))
        res_in.append(x)  #[batch,64,128,128]
        x = self.poolM_1(x)
        gram_in.append(x) #[batch,64,64,64]

        x = F.relu(self.convM_2(x))
        res_in.append(x)  #[batch,128,64,64]
        x = self.poolM_2(x)
        gram_in.append(x)  # [batch,128,32,32]

        x = F.relu(self.convM_3(x))
        res_in.append(x)  #[batch,256,32,32]
        x = self.poolM_3(x)
        gram_in.append(x)  # [batch,256,16,16]

        return x, res_in, gram_in

    def content_enc(self, x):
        """
        :param x: one picture of last K
        :return: result of VGG16 up to the third pooling layer, residual and indices of pool
        """
        res_in = []
        indices_c = []  #save indices of pool for unpool
        gram_in = []  # Gram matrix for motion encoder

        x = F.relu(self.convC1_1(x))
        x = F.relu(self.convC1_2(x))
        res_in.append(x)  #[batch,64,128,128]
        x, indices = self.poolC_1(x)
        indices_c.append(indices)
        gram_in.append(x)  # [batch,64,64,64]

        x = F.relu(self.convC2_1(x))
        x = F.relu(self.convC2_2(x))
        res_in.append(x)  #[batch,128,64,64]
        x, indices = self.poolC_2(x)
        indices_c.append(indices)
        gram_in.append(x)  # [batch,128,32,32]

        x = F.relu(self.convC3_1(x))
        x = F.relu(self.convC3_2(x))
        x = F.relu(self.convC3_3(x))
        res_in.append(x)  #[batch,256,32,32]
        x, indices = self.poolC_3(x)
        indices_c.append(indices)
        gram_in.append(x)  # [batch,256,16,16]

        return x, res_in, indices_c, gram_in

    def comb_layers(self, h_dyn, h_cont):
        """
        :param h_dyn: feature of motion encoder
        :param h_cont: feature of content encoder
        :return: combination result of three convolution
        """
        x = torch.cat((h_dyn, h_cont), 1)

        x = F.relu(self.convComb_1(x))
        x = F.relu(self.convComb_2(x))
        x = F.relu(self.convComb_3(x))

        return x

    def residual(self, res_m, res_c):
        """
        :param res_m: motion residual of three layer
        :param res_c: content residual of three layer
        :return: result of convolution
        """
        res_out = []

        x = torch.cat((res_m[0], res_c[0]), 1)
        x = F.relu(self.convRes1_1(x))
        x = self.convRes1_2(x)
        res_out.append(x)

        x = torch.cat((res_m[1], res_c[1]), 1)
        x = F.relu(self.convRes2_1(x))
        x = self.convRes2_2(x)
        res_out.append(x)

        x = torch.cat((res_m[2], res_c[2]), 1)
        x = F.relu(self.convRes3_1(x))
        x = self.convRes3_2(x)
        res_out.append(x)

        return res_out

    def content_dec(self, h_comb, res_connect, indices_c):
        """
        operator of decoder is contrary to content encoder
        :indices_c: pool indices of content encoder for unpool
        :param h_comb: result of combination layer
        :param res_connect: result of residual layer connect
        :return:
        """
        gram_in = []

        gram_in.append(h_comb)  # [batch,256,16,16]
        x = self.depool3(h_comb, indices_c[2]) # [batch, 256, 32, 32]
        x = torch.add(x, res_connect[2])
        x = F.relu(self.deconvC3_3(x))
        x = F.relu(self.deconvC3_2(x))
        x = F.relu(self.deconvC3_1(x))

        gram_in.append(x)  # [batch,128,32,32]
        x = self.depool2(x, indices_c[1]) # [batch, 128, 64, 64]
        x = torch.add(x, res_connect[1])
        x = F.relu(self.deconvC2_2(x))
        x = F.relu(self.deconvC2_1(x))

        gram_in.append(x)  # [batch,64,64,64]
        x = self.depool1(x, indices_c[0]) # [batch, 64, 128, 128]
        x = torch.add(x, res_connect[0])
        x = F.relu(self.deconvC1_2(x))
        x = F.tanh(self.deconvC1_1(x))

        return x, gram_in

