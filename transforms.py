#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:22:58 2017

@author: loop
"""
from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import types


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor(object):
    """
    Converts numpy.ndarray (N x H x W x C x 1) in the range
    [0, 255] to a torch.FloatTensor of shape (N x H x W x C x 1).
    """

    def __call__(self, pic):
        # handle numpy array
        img = torch.from_numpy(pic)
        # backard compability
        return img


class Normalize(object):
    """
    will normalize each channel of the torch.*Tensor, i.e.
    channel = channel/127.5 - 1
    """

    def __call__(self, tensor):
        # TODO: make efficient
        for t in tensor:
            t.div_(127.5).sub_(1)
        return tensor

class RandomHorizontalFlip(object):
    """
    Randomly horizontally flips the given numpy.ndarray
    (N x H x W x C x 1) with a probability of 0.5
    """

    def __call__(self, img):
        for n in xrange(img.shape[0]):
            if random.random() < 0.5:
                img[n] = img[n,:,::-1]
        return img


