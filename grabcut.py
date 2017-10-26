#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:31:35 2017

@author: ubuntu
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

sample_db_path = "./sample/"
test_db_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Tmp/LP/"
filename = "27.jpg"
fullpath = test_db_path + filename

img = cv2.imread(fullpath)
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (50,50,450,290)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img),plt.colorbar(),plt.show()