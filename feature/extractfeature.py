#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 06:40:47 2017

@author: ubuntu
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from feature.denoise import Denoise
import numpy

from misc.switch import switch
##
dataset_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Tmp/LP/"
filename = "11.jpg"
fullpath = dataset_path + filename
###
needNormalize = True
###
def extractROI(img):
    height, width, numChannels = img.shape
    x1 = int(width * 0.2)
    x2 = int(width * 0.8)
    y1 = int(height * 0.25)
    y2 = int(height * 0.75)
    return img[y1:y2,x1:x2,:]

def FeatureFiltering(pts,
                     ratio = 1.0,
                     filterRange = 20):

    refined = []#corners

    for pt in pts:
        count = 0
        for pt_ in pts:
            if numpy.linalg.norm(pt[0]-pt_[0]) < 30:
                count += 1
        if count > 5:
            refined.append(pt*ratio)
    return refined
    
def goodFeatures(gray,
                 numberoffeatures = 100,
                 qualityLevel = 0.005,
                 minDistance = 10):
    
    corners = cv2.goodFeaturesToTrack(gray,numberoffeatures,qualityLevel,minDistance)
    corners = np.int0(corners)
    
    return corners

def Laplacian(gray):
    denoised = cv2.GaussianBlur(gray,(5,5),0)
    laplacian = cv2.Laplacian(denoised,cv2.CV_64F, ksize = 3,scale = 2,delta = 1)
    laplacian -= np.amin(laplacian)
    laplacian = laplacian * 255 / (np.amax(laplacian) - np.amin(laplacian))
    laplacian[laplacian>255]=255
    laplacian = laplacian.astype('uint8')
    return laplacian

def FeatureSpace(img,
                 target="LP"):
    # RGB->HSV
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    #blur = cv2.GaussianBlur(s,(5,5),0)
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #s_ = np.array(s.shape)
    #s_ = cv2.normalize(s,  s_, 0, 255, cv2.NORM_MINMAX)
        
    for case in switch(target):
        if case('LP'):           
            return s
        if case('VIN'):
            return v           

def goodfeatures_revision(colr,isdebug=False):
    size=500.0
    h,w,c = colr.shape
    img = cv2.resize(colr,(int(w*size/h),int(size)))    
    fspace = Laplacian(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    corners =  goodFeatures(fspace)
    refined = FeatureFiltering(corners)
    if isdebug:
        checkFeatures(img,refined,isdebug)
    return refined
    
def refinedGoodFeatures(colr,
                        tgt,
                        model='LP',
                        Debug=False):
    size=500.0
    h,w,c = colr.shape
    img = cv2.resize(colr,(int(w*size/h),int(size)))
    img = Denoise(img)
    fspace = FeatureSpace(img,model)         # preprocess to get grayscale and threshold images
    corners =  goodFeatures(fspace)
    refined = FeatureFiltering(corners,ratio=tgt.shape[0]/size)
    if Debug:
        checkFeatures(tgt,refined,Debug)
        
    return refined

def SIFT(gray):

    sift = cv2.SIFT()
    kp = sift.detect(gray,None)
    return kp

def checkFeatures(img,
                  corners,
                  Debug=False):
    
    mark = img.copy()
    debug_img = img.copy()
    mark[mark>=0] = 0   
    for i in corners:
        x,y = i.ravel()
        cv2.circle(mark,(int(x),int(y)),1,(255,255,255),5)
        cv2.circle(debug_img,(int(x),int(y)),1,(255,0,0),5)

    if Debug:
        cv2.imshow("goodFeatures",debug_img)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
    
    return mark[:,:,0]

from feature.bbox import showResult
from skimage import io, morphology, img_as_bool, segmentation
from scipy import ndimage as ndi
from skimage import img_as_ubyte

def skeletonize(gray):
    out = ndi.distance_transform_edt(~gray)
    out = out < 0.05 * out.max()
    out = morphology.skeletonize(out)
    out = morphology.binary_dilation(out, morphology.selem.disk(1))
    out = segmentation.clear_border(out)
    out = out | gray
    return out

def morphological(img,
                  model=cv2.MORPH_OPEN):
    #http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
            #cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            #cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    for case in switch(model):
        if case(cv2.MORPH_OPEN):
            out=cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            break
        if case(cv2.MORPH_CLOSE):
            out=cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return out

def refinedCorners(img,
                   corners,
                   isdebug=False):
    #http://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html
    mark = img.copy()
    mark[mark>=0] = 0
    #img = cv2.drawKeypoints(img,corners)    
    for i in corners:
        x,y = i.ravel()
        cv2.circle(mark,(int(x),int(y)),1,(255,255,255),5)
    dilate = cv2.dilate(mark[:,:,0],(3,3),iterations=1)
    erode = cv2.dilate(dilate,(3,3),iterations=1)
    closing = morphological(erode,cv2.MORPH_CLOSE)
    erode = cv2.dilate(closing,(3,3),iterations=1)
    #erode = cv2.erode(closing,(3,3),iterations=1)
    #ret,thr = cv2.threshold(erode,0,255,cv2.THRESH_BINARY)
    binary = erode#mark[:,:,0]
    binary[binary>0]=1
    skeleton = img_as_ubyte(morphology.skeletonize_3d(binary))#skeletonize(binary))
    skeleton[skeleton > 0] = 255
    dilate = cv2.dilate(skeleton,(3,3))
    corners_ = np.transpose(np.nonzero(skeleton))
    cv_corners = []
    for corner in corners_:
        cv_corners.append([np.array([corner[1],corner[0]])])
    if isdebug:
        for pt in cv_corners:
            cv2.circle(img,(pt[0][0],pt[0][1]),1,(255,255,255),1)
        showResult("mark",mark)
        showResult("skeleton",skeleton)
        showResult("img",img)

    return cv_corners,mark[:,:,0]
    
def main():
    # Load
    origin = cv2.imread(fullpath)
    # Resize
    h,w,c = origin.shape
    img = cv2.resize(origin,((w*500)/h,500))
    # Extract ROI
    #img = extractROI(img)
    '''
    # Denoise
    img = Denoise(img)
    # Grayscale & Normalize
    if needNormalize == True:
        gray = FeatureSpace(img)         # preprocess to get grayscale and threshold images
    else:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Extract Features
    corners = goodFeatures(gray)
    # Refine Features
    refined =  FeatureFiltering(corners)
    # Show Result
    checkFeatures(img,refined)
    '''
    debug_img = cv2.resize(img,(0,0),fx = 0.4,fy = 0.4)
    refinedGoodFeatures(img,debug_img,Debug=True)
   
if __name__ == "__main__":
    main()