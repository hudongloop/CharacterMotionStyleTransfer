# -*- coding: utf-8 -*-
"""
create on Mon July 24
@author: loop
"""
import transforms
from utils import show_image, save_image
import KTH_training
import matplotlib.pyplot as plt
import torch.nn as nn

import numpy as np
import argparse
import os
import torch
from torch.autograd import Variable
import torch.optim as optim

#from mcnet import MCnet
#from loss import LOSS

def main(lr, batch_size, alpha, beta, image_size, K,
         T, num_iter, gpu, log_interval):

    # save and process folder
    prefix = ("training_kth_KTH_MCNET"
           + "_image_size="+str(image_size)
           + "_K="+str(K)
           + "_T="+str(T)
           + "_batch_size="+str(batch_size)
           + "_alpha="+str(alpha)
           + "_beta="+str(beta)
           + "_lr="+str(lr))

    print("\n"+prefix+"\n")
    checkpoint_dir = os.path.join("./models", prefix)
    samples_dir = os.path.join("./samples", prefix)
    summary_dir = os.path.join("./logs", prefix)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)


    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)

    # load data
    transform = transforms.Compose([#transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize()])
    train_set = KTH_training.KTH("../data/KTH/", batch_size, image_size, K, T,
                        transform=transform, shuffle=False)
    '''
    dataiter = iter(train_set)
    for i in range(20):
        # show data
        def imshow(img, pic):
            fig = plt.figure()
            npimg = img.numpy()[0,0,:,:,:]
            for i in xrange(10):
                fig.add_subplot(2, 5, i+1)
                img = (npimg[:,:,i] +1)*127.5
                plt.imshow(img, cmap='gray')
            plt.imshow((pic.numpy()[0,0,:,:]+1)*127.5, cmap='gray')
            plt.show()

        data, diff, pic= dataiter.next()
        imshow(data, pic)
    '''
    # network and loss
    #mcnet = MCnet()
    #loss = LOSS(lr, batch_size, alpha, beta, image_size, K, T, gpu)
    loss = torch.load("./logs/20180121KTH_MCNET_image_size=128_K=10_T=10_batch_size=2_alpha=1.0_beta=0.02_lr=1e-05/epoch200_weight.pt")
    if gpu:
        #mcnet.cuda()
        loss.cuda()

    save_loss = []
    # train
    for epoch in xrange(1) :
        dataiter = iter(train_set)
        len_dataiter = len(train_set) // batch_size
        #for batch_idx, (seq_batch, diff_batch) in enumerate(dataiter):
        for batch_idx, (seq_batch, diff_batch, pic_batch) in enumerate(dataiter):
            #if gpu:
            #    seq_batch, diff_batch = seq_batch.cuda(), diff_batch.cuda()
            #seq_batch, diff_batch = Variable(seq_batch), Variable(diff_batch)
            #output_list = mcnet(diff_batch, seq_batch[:, :, :, :, K-1])
            #output_seq = torch.cat(output_list, 4)

            #d_loss, g_loss, pre_data = loss(diff_batch, seq_batch)
            d_loss, g_loss, gram_loss, pre_data= loss(diff_batch, seq_batch, pic_batch, train=True)

            #show_image(pre_data, 2, 5)

            if batch_idx % log_interval == 0:
                print("Item:{} [{}/{}({:.1f}%)], D:{:.5f}, G:{:.5f}, M:{:.5f}".format(
                    epoch, batch_idx, len_dataiter,
                    100.*batch_idx/len_dataiter, d_loss.data[0], g_loss.data[0], gram_loss.data[0]))

                # save LOSS
                save_loss.append((d_loss.data[0], g_loss.data[0], gram_loss.data[0]))

                #print("Discriminator bias:{}, weight:{}".format(D_bias, D_wegiht))
                #print("L_img:{}, d_loss_gan:{}".format(L_img, d_loss_gan))

            if batch_idx % (log_interval) == 0:
                filename = os.path.join(samples_dir, "epoch_%dbatch_idx_%d.bmp" % (epoch, batch_idx))
                print("save generate img of " + filename)
                save_image(pre_data, 2, 5, filename)

                # save original data

                if epoch==0:
                    # save content image
                    contentname = os.path.join(samples_dir, "epoch_%dbatch_idx_%d_content.bmp" % (epoch, batch_idx))
                    print("save generate img of " + contentname)
                    save_image(pic_batch.unsqueeze_(4),1,1,contentname)

                    # save style image
                    stylename = os.path.join(samples_dir, "epoch_%dbatch_idx_%d_style.bmp" % (epoch, batch_idx))
                    print("save generate img of " + stylename)
                    save_image(seq_batch, 2, 5, stylename)

        if epoch % 50 == 0:
            print("save weight")
            torch.save(loss, (os.path.join(summary_dir, "epoch%d_weight.pt" % epoch)))
            print("save loss")
            np.savez_compressed(os.path.join(summary_dir, "epoch%d_loss.npz" % epoch), loss=save_loss)

    print('success')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MCnet Example")
    parser.add_argument("--lr", type=float, dest="lr",
                       default=0.00001, help="Base Learning Tate")
    parser.add_argument("--batch_size", type=int, dest="batch_size",
                        default=2, help="Mini-batch size")
    parser.add_argument("--alpha", type=float, dest="alpha",
                        default=1.0, help="Image loss weight")
    parser.add_argument("--beta", type=float, dest="beta",
                        default=0.02, help="GAN loss weight")
    parser.add_argument("--image_size", type=int, dest="image_size",
                        default=128, help="Mini-batch size")
    parser.add_argument("--K", type=int, dest="K",
                        default=10, help="Number of steps to observe from the past")
    parser.add_argument("--T", type=int, dest="T",
                        default=10, help="Number of steps into the future")
    parser.add_argument("--num_iter", type=int, dest="num_iter",
                        default=1305, help="Number of iterations")
    parser.add_argument("--gpu", action='store_true', default=True,
                        help='Use CUDA training')
    parser.add_argument("--log_interval", type=int, dest="log_interval",
                        default=1, help="How many interval to save log")

    args = parser.parse_args()
    main(**vars(args))