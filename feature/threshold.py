#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:43:03 2017

@author: ubuntu
"""

from skimage.filters import (threshold_otsu, threshold_niblack,threshold_yen,threshold_li,
                             threshold_sauvola)
from skimage import io, morphology, img_as_bool, segmentation
from skimage import img_as_ubyte
import numpy as np
import cv2

def windowthreshold(gray):
    h,w = gray.shape
    thr = np.zeros((gray.shape),dtype=np.uint8)    
    for ywin in range(int(h/4),h-1):
        if (ywin > 25 and ywin % 5 != 0) or ywin % 2 == 0:
            continue
        xwin = 2 * ywin
        for j in range(h/ywin+1):
            ymin = j * ywin
            if (h - j*ywin) < ywin:
                ymin = h - ywin           
            for i in range(w/xwin+1):
                xmin = i * xwin
                if (w - i*xwin) < xwin:
                    xmin = w - xwin
                src_cropped = gray[ymin:ymin+ywin,xmin:xmin+xwin]
                ret,binary = cv2.threshold(src_cropped,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                #thresh = threshold_otsu(src_cropped)#threshold_sauvola(src_cropped, window_size=ywin)#, k=0.8)
                #binary = src_cropped > thresh
                #binary = binary.astype(np.uint8) * 255
                dst_cropped = thr[ymin:ymin+ywin,xmin:xmin+xwin]
                thr[ymin:ymin+ywin,xmin:xmin+xwin] = cv2.max(dst_cropped.copy(),binary)
    #ret,thr = cv2.threshold(thr,0,255,cv2.THRESH_BINARY_INV)
    return thr

from feature.bbox import showResult

def compositeThreshold(gray,
                       mode='com'):
    if mode == 'otsu':
        otsu = threshold_otsu(gray)
        otsu_bin = gray > otsu
        otsu_bin = otsu_bin.astype(np.uint8) * 255
        return otsu_bin
    elif mode == 'yen':
        yen = threshold_yen(gray)
        yen_bin = gray > yen
        yen_bin = yen_bin.astype(np.uint8) * 255
        return yen_bin
    elif mode == 'li':
        li = threshold_li(gray)
        li_bin = gray > li
        li_bin = li_bin.astype(np.uint8) * 255
        return li_bin
    elif mode == 'niblack':
        niblack = threshold_niblack(gray,window_size=13, k=0.8)
        niblack_bin = gray > niblack
        niblack_bin = niblack_bin.astype(np.uint8) * 255
        return niblack_bin
    elif mode == 'sauvola':
        sauvola = threshold_sauvola(gray,window_size=13)
        sauvola_bin = gray > sauvola
        sauvola_bin = sauvola_bin.astype(np.uint8) * 255
        return sauvola_bin   
    elif mode == 'com':
        li = threshold_li(gray)
        li_bin = gray > li
        li_bin = li_bin.astype(np.uint8) * 255  
        otsu = threshold_otsu(gray)
        otsu_bin = gray > otsu
        otsu_bin = otsu_bin.astype(np.uint8) * 255
        yen = threshold_yen(gray)
        yen_bin = gray > yen
        yen_bin = yen_bin.astype(np.uint8) * 255
        return cv2.min(cv2.min(otsu_bin,li_bin),yen_bin)
    elif mode == "niblack-multi":
        thr = np.zeros((gray.shape),dtype=np.uint8)
        thr[thr>=0] = 255
        for k in np.linspace(-0.8, 0.2,5):#(-1.8,0.2,5)
            thresh_niblack = threshold_niblack(gray, window_size=25, k=k)
            binary_niblack = gray > thresh_niblack
            binary_niblack = binary_niblack.astype(np.uint8) * 255
            showResult("binary_niblack",binary_niblack)
            thr = cv2.min(thr,binary_niblack)
        return thr           
    else:
        sauvola = threshold_sauvola(gray, window_size=25, k=0.25)
        sauvola_bin = gray > sauvola
        sauvola_bin = sauvola_bin.astype(np.uint8) * 255
        niblack = threshold_niblack(gray,window_size=25, k=0.25)
        niblack_bin = gray > niblack
        niblack_bin = niblack_bin.astype(np.uint8) * 255
        return cv2.max(sauvola,niblack)
    #thr = cv2.max(li_bin,yen_bin)
    #return thr

###
def kojy_gray(image,
              isday=True):
    # blue mask + yellow mask
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = 255 - gray
    h,s,v = cv2.split(hsv)
    sat = (s.copy()).astype('float')
    val = (gray.copy()).astype('float')
    sat /= 255
    val /= 255
    #
    blue = h.copy()
    yellow = h.copy()
    if isday:
        blue = np.where((h<105)|(h>135)|(v<40)|(s<40),0,255)
        yellow = np.where((h<20)|(h>40),0,255)
    else:
        blue = np.where((h<105)|(h>135),0,255)
        yellow = np.where((h<20)|(h>40),0,255)        
    #
    hue = np.zeros(image.shape[:2])
    hue[hue>=0] = 0.1
    hue[blue>0] = 0.9
    hue[yellow>0] = 0.7
    if isday:
        '''
        mask= np.sqrt(val + sat)/5
        avg = np.average(mask)
        max_ = np.amax(mask)
        min_ = np.amin(mask)
        mask1= ((1-avg)+mask)**2
        mask1[mask1>1]=1
        mask2= ((1-(avg+min_)/2)+mask)**3
        mask2[mask2>1]=1
        mask2[mask2<1]=0
        mask3= ((1-(avg+max_)/2)+mask)**3
        mask3[mask3>1]=1
        mask = 0.3+2*(0.3+mask)**3
        '''
        mask = (sat*val)**2
        mask[mask>1]=1
        #mask2 = 
        #mask = 1 - mask
        return (mask*255).astype('uint8'),(np.sqrt(val*val)*255).astype('uint8')
    else:
        mask=(hue*sat)**3    
        return (mask*255).astype('uint8'),None

def smartthreshold(image,
                   isday=True):
    # blue mask + yellow mask
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = 255 - gray
    h,s,v = cv2.split(hsv)
    sat = (s.copy()).astype('float')
    val = (gray.copy()).astype('float')
    sat /= 255
    val /= 255
    #val = np.sqrt(val)
    #
    blue = h.copy()
    yellow = h.copy()
    if isday:
        blue = np.where((h<105)|(h>135)|(v<40)|(s<40),0,255)
        yellow = np.where((h<20)|(h>40),0,255)
    else:
        blue = np.where((h<105)|(h>135),0,255)
        yellow = np.where((h<20)|(h>40),0,255)        
    #
    hue = np.zeros(image.shape[:2])
    hue[hue>=0] = 0.5
    hue[blue>0] = 0.8
    hue[yellow>0] = 0.7
    if isday:
        mask= (val + np.sqrt(sat))/5
        avg = np.average(mask)
        #max_ = np.amax(mask)
        min_ = np.amin(mask)
        accumulate = np.zeros((mask.shape))
        for k in np.linspace(min_,avg,10):
            tmp = np.zeros((mask.shape))
            tmp= ((1-(avg+k)/2)+mask)**2
            tmp[tmp>1]=1
            tmp[tmp<1]=0
            accumulate += tmp
        mask = accumulate
        #np.minimum(mask1,mask2)
        #mask[mask2<1]=0
    else:
        mask=(hue*sat)**3
        
    return (mask*255).astype('uint8')    