# -*- coding: utf-8 -*-
"""
building LOSS of mcnet
Create on Wednesday August 3

@author: loop
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from mcnet import MCnet
from discriminator import Discriminator

# compute Gram Matrix
def gram_matrix(X):
    return X.mm(X.transpose(0,1))

def style_transfer(F, S):
    loss = []
    grad = []
    for l in xrange(len(S)):
        f = F[len(S)-l-1].view(F[len(S)-l-1].size(1),-1)
        s = S[l].view(S[l].size(1),-1)
        G_f = gram_matrix(f)
        G_s = gram_matrix(s)

        c = f.size(0)**-2 * f.size(1)**-2
        l = c/4.0 * (pow((G_f-G_s), 2).sum())

        loss.append(l)

    return (loss, grad)

def content_transfer(F, C):
    loss = []
    grad = []
    ratio = 0.0001
    for l in xrange(len(C)):
        f = F[len(C)-l-1].view(F[len(C)-l-1].size(1),-1)
        c = C[l].view(C[l].size(1),-1)

        l = 1.0/2.0 * (pow((f-c), 2).sum())

        loss.append(ratio*l)

    return (loss, grad)

class LOSS(nn.Module):
    def __init__(self, lr, batch_size, alpha, beta, image_size, K, T, gpu):
        super(LOSS, self).__init__()

        self.K = K
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size

        # define network and criterion
        self.mcnet = MCnet()
        self.discriminator = Discriminator(K, T)
        self.criterion = nn.BCELoss()

        # define value variable for training, it can convenient for multiple network
        self.true_data = torch.FloatTensor(batch_size, K+T, image_size, image_size)
        self.true_data_seq = torch.FloatTensor(batch_size, 1, image_size, image_size, K+T)
        self.fake_data_diff = torch.FloatTensor(batch_size, 1, image_size, image_size, K-1)
        self.fake_data_xt = torch.FloatTensor(batch_size, 1, image_size, image_size)
        self.label = torch.FloatTensor(batch_size)
        self.real_label = 1
        self.fake_label = 0

        if gpu:
            self.mcnet.cuda()
            self.discriminator.cuda()
            self.true_data = self.true_data.cuda()
            self.true_data_seq = self.true_data_seq.cuda()
            self.fake_data_diff = self.fake_data_diff.cuda()
            self.fake_data_xt = self.fake_data_xt.cuda()
            self.label = self.label.cuda()

        self.true_data = Variable(self.true_data)
        self.true_data_seq = Variable(self.true_data_seq)
        self.fake_data_diff = Variable(self.fake_data_diff)
        self.fake_data_xt = Variable(self.fake_data_xt)
        self.label = Variable(self.label)

        # define optimizer for each network to update weight
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr)
        self.optimizer_G = optim.Adam(self.mcnet.parameters(), lr)

    #def forward(self, diff_batch, seq_batch):
    def forward(self, diff_batch, seq_batch, pic_batch, train=True):
        """
        compute loss of Mcnet
        :param diff_batch: subtraction between of t and t-1 frame
        :param seq_batch: video sequence of T+K frame
        :return: discrimination loss and generation loss and predict value with cpu
        """
        if train:
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ############################

            self.discriminator.zero_grad()
            # train with real
            true_data_cpu = seq_batch.permute(0, 4, 2, 3, 1).contiguous()[:, :, :, :, 0] # sequence as channel [batch,seq+channel,H,W]
            self.true_data.data.resize_(true_data_cpu.size()).copy_(true_data_cpu) # copy data as Variable with gpu
            self.label.data.resize_(self.batch_size).fill_(self.real_label) # copy label as Variable with gpu

            self.true_data = self.true_data[:, 0:self.K, :, :] # discriminator is first K frame

            true_dis = self.discriminator(self.true_data) # truth data for discriminator
            d_loss_real = self.criterion(true_dis, self.label) # cross entropy for criterion
            d_loss_real.backward() # computer gradient

            # train with fake
            #xt_cpu = seq_batch[:, :, :, :, self.K - 1] # picture of last frame
            xt_cpu = pic_batch
            self.fake_data_diff.data.resize_(diff_batch.size()).copy_(diff_batch)# copy diff data as Variable with gpu
            self.fake_data_xt.data.resize_(xt_cpu.size()).copy_(xt_cpu) # copy last frame data as Variable with gpu
            self.true_data_seq.data.resize_(seq_batch.size()).copy_(seq_batch) # copy seq data as Variable with gpu

            output_list, gram = self.mcnet(self.fake_data_diff, self.fake_data_xt) # generate data of Mcnet
            predict = torch.cat(output_list, 4) # concatenate gen data of T seq [batch,channel,H,W,seq]
            gen_data = torch.cat([self.true_data_seq[:, :, :, :, :self.K], predict], # concatenate prior K data and sequence as channel
                                 4).permute(0, 4, 2, 3, 1).contiguous()[:, :, :, :, 0]
            self.label.data.fill_(self.fake_label)

            gen_data = gen_data[:, self.K:self.K+self.T, :, :] # discriminator is first K frame

            gen_dis = self.discriminator(gen_data.detach())
            d_loss_fake = self.criterion(gen_dis, self.label)
            d_loss_fake.backward()

            self.optimizer_D.step() # Adam update weight
            d_loss = d_loss_fake + d_loss_real # discrimination loss

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.mcnet.zero_grad()
            self.label.data.fill_(self.real_label)
            gen_dis = self.discriminator(gen_data)
            d_loss_gan = self.criterion(gen_dis, self.label)

            #L_img = self.loss_img(self.true_data_seq, predict) # compute L_img

            # (3) Gram Matrix Loss
            gram_loss = self.loss_gram(gram)

            #g_loss = self.alpha * L_img + self.beta * d_loss_gan # generation loss
            g_loss = self.alpha*gram_loss  + self.beta*d_loss_gan # generation loss
            g_loss.backward()

            self.optimizer_G.step()

            # show the parameter of Discriminator
            #D_bias = self.discriminator.conv1.bias.data[0:3].cpu().view(1,-1).numpy()
            #D_weight = self.discriminator.conv1.weight.data[0,0,0,0:3].cpu().view(1,-1).numpy()

            #return d_loss, g_loss, predict.data.cpu()
            return d_loss, g_loss, gram_loss, predict.data.cpu()
        else:
            xt_cpu = pic_batch
            self.fake_data_diff.data.resize_(diff_batch.size()).copy_(diff_batch)  # copy diff data as Variable with gpu
            self.fake_data_xt.data.resize_(xt_cpu.size()).copy_(xt_cpu)  # copy last frame data as Variable with gpu
            self.true_data_seq.data.resize_(seq_batch.size()).copy_(seq_batch)  # copy seq data as Variable with gpu

            output_list, gram = self.mcnet(self.fake_data_diff, self.fake_data_xt)  # generate data of Mcnet
            predict = torch.cat(output_list, 4)  # concatenate gen data of T seq [batch,channel,H,W,seq]

            return predict

    def loss_img(self, target, predict):
        # convert data to gray with 3 channel and combine batch and sequence
        true_sim = target[:, :, :, :, self.K:].add(1.0).div(2.0)
        #true_sim = target[:, :, :, :, :self.K].add(1.0).div(2.0)
        true_sim = true_sim.repeat(1, 3, 1, 1, 1).permute(0, 4, 1, 2, 3).contiguous()

        true_sim = true_sim.view(-1,
                                       true_sim.size(2),
                                       true_sim.size(3),
                                       true_sim.size(4))

        gen_sim = predict.add(1.0).div(2.0)
        gen_sim = gen_sim.repeat(1, 3, 1, 1, 1).permute(0, 4, 1, 2, 3).contiguous()
        gen_sim = gen_sim.view(-1,
                                     gen_sim.size(2),
                                     gen_sim.size(3),
                                     gen_sim.size(4))

        loss_p = self.loss_p(target[:, :, :, :, self.K:], predict, 2.0)
        #loss_p = self.loss_p(target[:, :, :, :, :self.K], predict, 2.0)
        loss_gld = self.loss_gld(true_sim, gen_sim, 1.0)
        L_img = loss_p + loss_gld

        return  L_img

    def loss_p(self, tar, pre, p):
        """
        loss_p = mean(||tar - pre||_2^2)
        :param tar: ground truth value
        :param pre: predict value
        :p: hyper-parameters of loss_p
        :return: loss_p
        """
        return torch.mean((pre - tar) ** p)

    def loss_gld(self, tar, pre, alpha):
        """
        match the gradients of such pixel values
        mean(|(|y_{i,j}-y_{i-1,j}| - |z_{i,j}-z_{i-1,j}|)|^n +
        |(|y_{i,j-1}-y_{i,j}| - |z_{i,j-1}-z_{i,j}|)|^n)

        :param tar: ground truth value
        :param pre: predict value
        :alpha: hyper-parameters of loss_gld
        :return: loss_gld
        """
        pos = torch.eye(3)
        neg = -1 * pos

        # weight for conv is [out_channel,in_channel,kH,kW]
        # subtraction between center and left
        weight_x = torch.zeros([3, 3, 1, 2])
        weight_x[:, :, 0, 0] = neg
        weight_x[:, :, 0, 1] = pos

        # subtraction between center and up
        weight_y = torch.zeros([3, 3, 2, 1])
        weight_y[:, :, 0, 0] = pos
        weight_y[:, :, 1, 0] = neg

        weight_x = Variable(weight_x.cuda())
        weight_y = Variable(weight_y.cuda())

        gen_dx = torch.abs(F.conv2d(pre, weight_x, padding=1))
        gen_dy = torch.abs(F.conv2d(pre, weight_y, padding=1))
        true_dx = torch.abs(F.conv2d(tar, weight_x, padding=1))
        true_dy = torch.abs(F.conv2d(tar, weight_y, padding=1))

        grad_diff_x = torch.abs(true_dx - gen_dx)
        grad_diff_y = torch.abs(true_dy - gen_dy)

        return torch.mean(grad_diff_x ** alpha + grad_diff_y ** alpha)

    def loss_gram(self, gram):

        loss = 0.0
        for t in xrange(len(gram)):
            # gram_s,gram_C:top to bottom. gram_f bottom to top
            (gram_s, gram_c, gram_f) = gram[t]

            # loss and grad of style
            (loss_s, grad_s) = style_transfer(gram_f, gram_s)

            # loss and grad of content
            (loss_c, grad_c) = content_transfer(gram_f, gram_c)

            for l in xrange(len(loss_s)):
                loss = loss + loss_s[l] + loss_c[l]

        loss = loss/len(gram)

        return loss
