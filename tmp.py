#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 05:06:47 2017

@author: junying
"""
##
sample_db_path = "./sample/"
test_db_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Tmp/VIN/"
filename = "1.jpg"
fullpath = test_db_path + filename
###

import cv2
import numpy as np
import tesserocr
from PIL import Image

def opencv2pillow(image):
    return Image.fromarray(image)
print tesserocr.tesseract_version()  # print tesseract-ocr version
print tesserocr.get_languages()
image = cv2.imread(fullpath)
image = opencv2pillow(image)
print tesserocr.image_to_text(image)