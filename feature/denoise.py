#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 07:00:54 2017

@author: ubuntu
"""

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2

def Denoise(img):
    return cv2.fastNlMeansDenoisingColored(img,None,7,7,5,15)

def resize(img):
    size=list(img.size)
    size[0] /= 2
    size[1] /= 2
    resized = img.resize(size, Image.NEAREST)
    return resized