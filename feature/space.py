#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 21:55:24 2017

@author: @jy
"""
import types
import cv2
import numpy as np
from misc.preprocess import maximizeContrast as icontrast

def uint8tobinary(mask,reverse=False):
    mask_ = mask.copy()
    if reverse:
        mask_[mask>128] = 1
        mask_[mask<=128] = 0
    else:
        mask_[mask>128] = 0
        mask_[mask<=128] = 1
    return mask_

def maskize(img,masks,reverse=False):
    if type(masks) is types.ListType:
        mask_ = uint8tobinary(masks[0],reverse=reverse)
        for i in range(1,len(masks)):
            mask_ *= uint8tobinary(masks[i])
    else:    
        mask_ = uint8tobinary(masks,reverse=reverse)
    if len(img.shape) > 2:
        masked = img*mask_[:,:,np.newaxis]
    else:
        masked = img*mask_
    return masked

def Laplacian(gray,needcontrast=True):
    denoised = cv2.GaussianBlur(gray,(5,5),0)
    laplacian = cv2.Laplacian(denoised,cv2.CV_64F, ksize = 3,scale = 2,delta = 1)
    laplacian = NormalizeEx(laplacian)
    if needcontrast:
        return icontrast(laplacian)#laplacian
    else:
        return laplacian

def NormalizeEx(mask):
    mask -= np.amin(mask)
    max_,min_ = np.amax(mask),np.amin(mask)
    mask = mask * 255 / (max_ - min_) if max_ != min_ else mask * 255
    mask[mask>255]=255
    mask = mask.astype('uint8')
    return mask
    
def Normalize(mask):
    min_,max_ = np.amin(mask),np.amax(mask)
    mask -= min_
    mask = mask * 1.0 / (max_ - min_)
    return mask

def NormalizeT(image):
    norm = np.zeros((image.shape),np.float32)
    norm_rgb = np.zeros((image.shape),np.uint8)
    b = image[:,:,0]
    g = image[:,:,1]
    r = image[:,:,2]
    
    sum_ = b.astype("float") + g.astype("float") + r.astype("float")
    sum_ += 1
    
    norm[:,:,0] = b/sum_
    norm[:,:,1] = g/sum_
    norm[:,:,2] = r/sum_
    norm *= 255.0
    
    norm_rgb = cv2.convertScaleAbs(norm)
    
    return norm_rgb

def NormalizedCV(image):
    normalized = np.zeros((image.shape),np.uint8)
    normalized = cv2.normalize(image,normalized, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return normalized

def Sobel(gray):
    sobel_vertical = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize = 3)
    sobel_horizontal = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    sobel = cv2.min(sobel_horizontal,sobel_vertical)
    #print np.amin(sobel),np.amax(sobel)
    min_,max_ = np.amin(sobel),np.amax(sobel)
    sobel -= min_
    sobel = sobel * 255 / (max_ - min_)
    sobel = sobel.astype('uint8')
    return sobel    

def Garbor(gray):        
    #https://corpocrat.com/2015/03/25/applying-gabor-filter-on-faces-using-opencv/
    src = gray.astype('float32')
    ksize,sigma,gamma,ps = 31, 1.0, 0.02, 0
    filters = []
    for theta in np.arange(0, np.pi, np.pi / 8):
        for lamda in np.arange(0, np.pi, np.pi/4): 
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, ps, ktype=cv2.CV_32F)
            kernel /= 1.5*kernel.sum()
            filters.append(kernel)
    dest = np.zeros((src.shape),np.float32)#np.zeros_like(src.shape)
    for kernel in filters:
        fimg = cv2.filter2D(src, cv2.CV_32F, kernel)
        np.maximum(dest, fimg, dest)
    #ksize,sigma,theta,lamda,gamma,ps = 31, 1, 0, 1.0, 0.02, 0
    #kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, ps, ktype=cv2.CV_32F)
    return dest.astype('uint8')    

def DoG(gray):    
    #equalized_image = cv2.equalizeHist(gray)
    imgb1 = cv2.GaussianBlur(gray, (11, 11), 0)
    imgb2 = cv2.GaussianBlur(gray, (31, 31), 0)
    return imgb1 - imgb2#Difference of Gaussians    

from feature.bbox import showResult

def tophatblackhat(gray):
    gray = cv2.GaussianBlur(gray,(3,3),0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    showResult("tophat",tophat)
    showResult("blackhat",blackhat)

def tophatmask(gray):
    gray = cv2.GaussianBlur(gray,(3,3),0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    return NormalizeEx(tophat)

def close(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    closed=cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return closed

def TopHat(gray):
    # initialize a rectangular (wider than it is tall) and square
    # structuring kernel
    h,w = gray.shape
    #gray=255-gray
    gray = cv2.resize(gray,None,fx=0.4,fy=0.4)
    #gray = cv2.GaussianBlur(gray,(3,3),0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # apply a tophat (whitehat) morphological operator to find light
    # regions against a dark background (i.e., the credit card numbers)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    hat = cv2.max(tophat,blackhat)
    #showResult("tophat",tophat)
    # compute the Scharr gradient of the tophat image, then scale
    # the rest back into the range [0, 255]
    gradX = cv2.Sobel(hat, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)
    #gradY = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=0, dy=1,ksize=-1)
    grad = gradX#cv2.min(gradX,gradY)
    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    grad = (255 * ((grad - minVal) / (maxVal - minVal)))
    grad = grad.astype("uint8")
    # apply a closing operation using the rectangular kernel to help
    # cloes gaps in between credit card number digits, then apply
    # Otsu's thresholding method to binarize the image
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]    
    # apply a second closing operation to the binary image, again
    # to help close gaps between credit card number regions
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, sqKernel)
    thresh = cv2.dilate(thresh,None, iterations=6)
    thresh = cv2.resize(thresh,(w,h),interpolation=cv2.INTER_LINEAR)
    return thresh

ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

def AdaptiveThreshold(gray):
    return cv2.adaptiveThreshold(gray, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
#from skimage import data
#from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
    
def Entropy(gray):
    en_ = entropy(gray, disk(3))
    min_ = np.amin(en_)
    max_ = np.amax(en_)
    en_ -= min_
    en_ = en_ * 255 / (max_ - min_)
    en_ = en_.astype('uint8')
    return en_