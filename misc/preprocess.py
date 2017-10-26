#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 05:28:35 2017

@author: ubuntu
"""

# Preprocess.py

import cv2
import numpy as np
# module level variables ##########################################################################
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

###################################################################################################
def preprocess(origin):
    # Colr2Gray
    fspace = colr2gray(origin)#Denoise(origin))#EqualizeHist(Denoise(origin),cv2.COLOR_YCR_CB2BGR))
    # Erode
    cv2.erode(fspace,(3,3),fspace)
    # Contrast
    contrasted = maximizeContrast(fspace)
    h, w = fspace.shape
    # Blur
    blurred = np.zeros(fspace.shape, np.uint8)
    blurred = cv2.GaussianBlur(contrasted, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    # Threshold
    #ret, thr = cv2.threshold(imgBlurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thr = cv2.adaptiveThreshold(blurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)    
    #erode = np.zeros(fspace.shape, np.uint8)
    

    return fspace, thr
# end function

###################################################################################################
def colr2gray(origin):
    height, width, channels = origin.shape
    hsv = np.zeros((origin.shape), np.uint8)
    hsv = cv2.cvtColor(origin, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v[:,:] = 255 - v[:,:]
    #v[v==128] = 128
    #v[v>128] = 255 - v[v>128]
    #v[v<128] = 255 - v[v<128]
    #s_ = cv2.normalize(imgValue,  normalizedImg, 0, 255, cv2.NORM_MINMAX)

    return v
# end function

###################################################################################################
def maximizeContrast(imgGrayscale):

    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat
# end function
