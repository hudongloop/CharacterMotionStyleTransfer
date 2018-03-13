#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:22:58 2017

@author: loop
"""
from __future__ import print_function
import torch.utils.data as data
import torch

import os
import numpy as np
import random
import imageio
import cv2
import joblib

# proces data with parallel, must outside of class
def load_kth_data(f_name, data_path, image_size, L):
    """
    :param f_name: video name
    :param data_path: data path
    :param image_size: image size
    :param L: extract L frame of video
    :return: sequence frame of K+T len
    """

    tokens = f_name.split()
    vid_path = os.path.join(data_path, tokens[0] + "_uncomp.avi")
    vid = imageio.get_reader(vid_path, "ffmpeg")  # load video
    low = int(tokens[1])  # start of video
    # make sure the len of video is than L
    high = np.min([int(tokens[2]), vid.get_length()]) - L + 1

    # the len of video is equal L
    if (low == high):
        stidx = 0
    else:
        # the len of video is less-than L, print video path and the error for next line
        if (low >= high): print(vid_path)
        # the len of video greater than L, and the start is random of low-high
        stidx = np.random.randint(low=low, high=high)

    # extract video of L len [in_channel, image_w, image_h, sequence]
    seq = np.zeros((1, image_size, image_size, L), dtype="float32")
    for t in xrange(L):
        img = cv2.cvtColor(cv2.resize(vid.get_data(stidx + t), (image_size, image_size)),
                           cv2.COLOR_RGB2GRAY)
        seq[0, :, :, t] = img[:, :]

    return seq

# load content picture of KTH data
def load_kth_picture(f_name, data_path, image_size):
    """
    :param f_name: video name
    :param data_path: data path
    :param image_size: image size
    :return: sequence frame of K+T len
    """

    tokens = f_name.split()
    vid_path = os.path.join(data_path, tokens[0] + "_uncomp.avi")
    vid = imageio.get_reader(vid_path, "ffmpeg")  # load video
    low = int(tokens[1])  # start of video

    picture = np.zeros((1, image_size, image_size), dtype="float32")
    img = cv2.cvtColor(cv2.resize(vid.get_data(low), (image_size, image_size)),
                    cv2.COLOR_RGB2GRAY)
    picture[0, :, :] = img[:, :]

    return picture

class KTH(data.Dataset):

    train_file_dir = "train_data_list_trimmed.txt"
    test_file_dir = ""

    def __init__(self, root, batch_size, image_size, K, T,
                 train = True, transform=None, shuffle=True):
        self.root = root
        self.train = train
        self.transform = transform
        self.batch_size = batch_size
        self.image_size = image_size
        self.K = K
        self.T = T

        self.trainFiles = ""
        self.testFiles = ""
        self.mini_batches = ""

        if self.train:
            with open(os.path.join(root, self.train_file_dir), "r") as f:
                self.trainFiles = f.readlines()
            self.mini_batches = self.get_minibatches_idx(len(self.trainFiles), self.batch_size, shuffle=shuffle)

    def __getitem__(self, index):

        # read video data of mini-batch with parallel method
        Ls = np.repeat(np.array([self.T + self.K]), self.batch_size, axis=0) # video length of past and feature
        paths = np.repeat(self.root, self.batch_size, axis=0)
        files = np.array(self.trainFiles)[self.mini_batches[index][1]]
        shapes = np.repeat(np.array([self.image_size]), self.batch_size, axis=0)

        with joblib.Parallel(n_jobs=self.batch_size) as parallel:
            output = parallel(joblib.delayed(load_kth_data)(f, p, img_size, l)
                                                                for f, p, img_size, l in zip(files,
                                                                                                paths,
                                                                                                shapes,
                                                                                                Ls))
        # save batch data 1 is in_channel
        seq_batch = np.zeros((self.batch_size, 1, self.image_size, self.image_size,
                             self.K + self.T), dtype="float32")
        for i in xrange(self.batch_size):
            seq_batch[i] = output[i]


        # load picture of KTH as content. only one batch
        pic_batch = np.zeros((self.batch_size, 1, self.image_size, self.image_size),
                             dtype="float32")
        pic = load_kth_picture(self.trainFiles[index],self.root,self.image_size)
        pic_batch[0] = pic


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.transform is not None:
            seq_batch = self.transform(seq_batch)

            pic_batch = self.transform(pic_batch)

        # compute subtraction between t and t-1
        diff_batch = torch.zeros(self.batch_size, 1, self.image_size, self.image_size, self.K-1)
        for t in xrange(1, self.K):
            previous = seq_batch[:, :, :, :, t-1].add(1.0).div(2.0) # convert gray image[0-1]
            current =  seq_batch[:, :, :, :, t].add(1.0).div(2.0)
            diff_batch[:, :, :, :, t-1] = current.sub(previous)

        # [batch, channel, H, W, sequence]
        #return seq_batch, diff_batch
        return seq_batch, diff_batch, pic_batch


    def __len__(self):
        if self.train:
            return len(self.trainFiles)
        else:
            return len(self.testFiles)

    def get_minibatches_idx(self, n, minibatch_size, shuffle=False):
        """
        :param n: len of data
        :param minibatch_size: minibatch size of data
        :param shuffle: shuffle the data
        :return: len of minibatches and minibatches
        """

        idx_list = np.arange(n, dtype="int32")

        # shuffle
        if shuffle:
            random.shuffle(idx_list) # also use torch.randperm()

        # segment
        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:
                                        minibatch_start + minibatch_size])
            minibatch_start += minibatch_size

        # processing the last batch
        if (minibatch_start != n):
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)

