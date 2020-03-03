#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:39:32 2017

@author: junying

ref:https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_table_of_contents_contours/py_table_of_contents_contours.html?highlight=convex
"""
import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed,find_boundaries,clear_border
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage import data, segmentation, filters, color
from skimage.future import graph

import cv2

from feature.denoise import Denoise
from feature.colorspace import rgb2hsv,opencv2skimage,skimage2opencv,checkBlue,checkYellow,equalizehist
from feature.extractfeature import refinedGoodFeatures,checkFeatures,FeatureSpace,refinedCorners
from feature.bbox import findBBox,drawBBox,resizeBBoxes,BBoxes2ROIs,showResult,contour2bbox,close
import time

####################################################################################
################################QuickShift-Based####################################
####################################################################################
def drawLabels(img,
               labels,
               Debug=False):
    out = color.label2rgb(labels, img, kind='avg')
    if Debug:
        print('level1 segments: {}'.format(len(np.unique(labels))))
        #mark = segmentation.mark_boundaries(out, labels, (1, 0, 0))
        fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 12))
        
        ax[0].imshow(img)
        ax[1].imshow(out)
        #ax[2].imshow(mark)
        
        for a in ax:
            a.axis('off')
        
        plt.tight_layout()
        
    return out
##########################################################
# Image2Labels based on Segmentation
##########################################################
def seg(img,
        Debug=False):
    # Level 1
    start = time.time() * 1000
    Level = "QuickShift"
    if Level == "QuickShift":    
        labels = segmentation.quickshift(img, kernel_size=3, max_dist=10, ratio=1.0)#.slic(img, compactness=30, n_segments=400)
    elif Level == "SLIC":
        labels = segmentation.slic(img, compactness=30, n_segments=400)
    elif Level == "felzenszwalb":
        labels = segmentation.felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    elif Level == "Watershed":
        gradient = sobel(rgb2gray(img))
        labels = segmentation.watershed(gradient, markers=250, compactness=0.001)    
    out = color.label2rgb(labels, img, kind='avg')
    # Show Result
    if Debug:
        print("level1 took ",int(time.time() * 1000 - start),"ms")
        drawLabels(img,labels,Debug)
    return out,labels

#############################################
# This function refines lables with color info and corners
# Leave only labels with specific color and corners
# Background is marked as zero
#############################################
def refineLabels(out,
                 labels,
                 corners,
                 howmany=3):
    # pick up labels on corners
    containcorners = []
    for corner in corners:
        x = int(corner[0][0])
        y = int(corner[0][1])
        hsv = rgb2hsv(out[y,x])
        # accept only if candidate color(yellow or blue)
        if checkBlue(hsv) or checkYellow(hsv):
            containcorners.append(labels[y][x])
    # make table(corners,label)
    match = []
    for label in np.unique(containcorners):
        match.append([(containcorners == label).sum(),label])
        
    match = sorted(match,reverse=True)
    match = np.array(match)
    # refine howmany
    howmany = howmany if howmany < match.shape[0] else match.shape[0]
    # select labels with enough corners    
    pickup = []
    for i in range(howmany):
        pickup.append(match[i][1])
    # mark bg with 0
    mark = 0
    labels_ = np.zeros(labels.shape)
    #
    for label in np.unique(pickup):
        mark += 1
        labels_[labels == label] = mark
    # make saturation
    
    return labels_

GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)
ADAPTIVE_THRESH_BLOCK_SIZE = 7
ADAPTIVE_THRESH_WEIGHT = 3

##########################################################
# change the color of non-meaningful region with specific color
#########################################################
def changebgcolr(img,
                 labels,
                 colr=(0,0,0),
                 Debug=False):
    h,w,c = img.shape
    for j in range(h):
        for i in range(w):
            if labels[j][i] == 0:
                for k in range(3):
                    img[j][i][k] = colr[k]
    if Debug:
        cv2.imshow("img",skimage2opencv(img))
        cv2.waitKey(10000)
        cv2.destroyAllWindows()

# Label2Contour
def label2contour(label):
    #http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.html
    boundary = find_boundaries(label, mode='inner').astype(np.uint8)
    contour = np.transpose(np.nonzero(boundary))
    tmp = contour[:,1].copy()
    contour[:,1] = contour[:,0]
    contour[:,0] = tmp
    bbox = contour2bbox(contour)
    return bbox
# Labels2Contour
def labels2contours(labels):
    contours = []
    for label in np.unique(labels):
        tmp = labels.copy()
        tmp[tmp != label] = 0
        contour = label2contour(tmp)
        contours.append(contour)
    return contours
# Labels2Boundaries
def labels2boundaries(labels):
    #http://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.find_boundaries
    cleared = clear_border(labels)
    boundary = find_boundaries(cleared, mode='inner').astype(np.uint8)
    boundary[boundary != 0] = 255
    showResult("boundaries",boundary)
    
##########################################################
# Image2BBoxes
##########################################################        
def detect_by_seg_gf(origin,
                     isdebug=False):
    if origin is None:
        return None
    # Default Size
    h,w,c = origin.shape
    size = 200.0
    #origin = opencv2skimage(origin)
    # Resize
    img = cv2.resize(origin,(int(w*size/h),int(size)))
    # Blur
    blur = cv2.GaussianBlur(img,(5,5),3)
    # Equalization Hist
    #origin = equalizehist(origin)
    
    # Extract Good Features
    corners = refinedGoodFeatures(origin,img,
                                  model='LP')
    corners_,handwrite =  refinedCorners(img,corners,False)
    checkFeatures(img,corners,True)
    # Opencv2Skimage
    skimg = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
    # Segmentation
    out,labels = seg(skimg,Debug=False)
    # Eval Label
    labels = refineLabels(out,labels,corners_,howmany=20)
    # Show Result
    out = drawLabels(skimg,labels,Debug=isdebug)
    if out is None:
        return None
    changebgcolr(out,labels)
    #showResult("labelout",skimage2opencv(out))
    # Find Candidate 
    bboxes = findBBox(img,out,
                      model='LP',
                      debug=isdebug)
    # Resize
    if bboxes is not None:
        bboxes = resizeBBoxes(bboxes,h/size)
    # Check Candidate
    if isdebug and bboxes is not None:
        drawBBox(origin,bboxes,debug=isdebug)
    # Crop Rois
    rois = BBoxes2ROIs(origin,bboxes)
    if isdebug and rois is not None:
        for i in range(len(rois)):
            showResult("cropped",rois[i])
    '''
    if isdebug:
        #labels2boundaries(labels)
        contours = labels2contours(labels)
        drawBBox(img,contours,debug=True)
    '''
    return bboxes,rois
####################################################################################
################################GrabCut-Based##################################
####################################################################################
def colormask(image,
              isday=True,
              isdebug=False):
    # blue mask + yellow mask
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    sat = (s.copy()).astype('float')
    val = (v.copy()).astype('float')
    sat /= 255
    val /= 255
    #
    blue = h.copy()
    yellow = h.copy()
    if isday:
        blue = np.where((h<100)|(h>135)|(v<40)|(s<40),0,255)
        yellow = np.where((h<20)|(h>40)|(v<40)|(s<40),0,255)
    else:
        blue = np.where((h<105)|(h>135),0,255)
        yellow = np.where((h<20)|(h>40),0,255)        
    #
    hue = np.zeros(image.shape[:2])
    hue[hue>=0] = 0.2
    hue[blue>0] = 0.9
    hue[yellow>0] = 0.7
    if isday:
        mask=(hue*sat*val)**2
    else:
        mask=(hue*sat)**3
    #
    if isdebug:
        showResult("colormask:mask",mask*255)
        
    return mask

def featuremask(gf_img):
    mask = np.zeros(gf_img.shape[:2])
    mask[gf_img==255] = 0.8
    mask[gf_img<255] = 0.17
    return mask

def regionmask(image):
    mask = np.zeros(image.shape[:2])
    mask[mask>=0] = 1.0
    h,w = mask.shape
    mask[0:int(0.25*h),:] = 0
    mask[int(0.98*h):h-1,:] = 0
    mask[:,0:int(0.02*w)] = 0
    mask[:,int(0.98*w):w-1] = 0
    return mask

def mkfinalmask(image,
                gf_image=None,
                isday=True,
                isdebug=False):
    colrmask = colormask(image,isday)
    regmask = regionmask(image)
    mask = colrmask * regmask
    if gf_image is not None:
        gfmask = featuremask(gf_image)
        mask *= gfmask
        
    mask = (mask * 255).astype('uint8')
    if isdebug:
        showResult("mkfinalmask:mask",mask*255)
    return mask

import random
from feature.bbox import resizeBBox

def refine_gfimage(img,mask):
    # Draw BBox using gf
    bboxes = findBBox(img,mask)
    if bboxes is None:
        return mask

    box_img = np.zeros((img.shape),dtype=np.uint8)
    #cv2.drawContours(box_img,[bboxes],-1,(255,255,255),1)
    for bbox in bboxes:
        bbox_ = resizeBBox(bbox,ratio=1.0)
        cv2.drawContours(box_img,[bbox_],0,(255,255,255),1)
    newmask = cv2.cvtColor(box_img,cv2.COLOR_BGR2GRAY)  
    #
    img_contours,contours,hierarchy = cv2.findContours(newmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    img_contours = np.zeros((img.shape),dtype=np.uint8)
    if len(contours) == 0:
        return mask
    for contour in contours:
        random.shuffle(contour)
        cv2.drawContours(img_contours,[contour],0,(255,255,255),1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    newmask=cv2.morphologyEx(img_contours[:,:,0], cv2.MORPH_CLOSE, kernel)
    return newmask

def dayornight(img):
    h,w,c = img.shape
    avg = np.sum(img) / (w*h*c)
    if avg > 70:
        return True
    else:
        return False

from feature.threshold import compositeThreshold

def detect_by_probability(origin,
                 isdebug=False):
    if origin is None:
        return None
    # Default Size
    h,w,c = origin.shape
    size = 200.0
    # Resize
    img = cv2.resize(origin,(int(w*size/h),int(size)))
    #showResult("img",img)
    #  
    if dayornight(img):
        # Extract Good Features
        corners = refinedGoodFeatures(origin,img)
        mask = checkFeatures(img,corners,False)
        closing=close(mask)
        refined_gfmask = refine_gfimage(img,closing)
        #showResult("refined_gfmask",refined_gfmask)
        finalmask = mkfinalmask(img,refined_gfmask,isday=True)
    else:
        finalmask = mkfinalmask(img,None,isday=False)
    #
    #showResult("masktest",finalmask)
    #ret,binary = cv2.threshold(finalmask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary = compositeThreshold(finalmask,mode='otsu')
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    closing=cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if isdebug:
        showResult("masktest",closing)
    # Find Candidate
    bboxes = findBBox(img,closing,isdebug=True)
    # Resize
    if bboxes is not None:
        bboxes = resizeBBoxes(bboxes,h/size)
    # Check Result
    if isdebug and bboxes is not None:
        drawBBox(origin,bboxes,debug=True)
    # Crop Rois
    rois = BBoxes2ROIs(origin,bboxes)
    if isdebug and rois is not None:
        for i in range(len(rois)):
            showResult("cropped",rois[i])
            
    return bboxes,rois
####################################################################################
################################GoodFeatures-Based##################################
####################################################################################
    
# Image2BBoxes
def detect_by_gf(origin,
                 isdebug=False):
    if origin is None:
        return None
    # Default Size
    h,w,c = origin.shape
    size = 200.0
    # Resize
    img = cv2.resize(origin,(int(w*size/h),int(size)))   
    # Extract Good Features
    corners = refinedGoodFeatures(origin,img,
                                  model='LP')
    mask = checkFeatures(img,corners,True)
    # Find Candidate
    bboxes = findBBox(img,mask,
                      model='LP',
                      debug=True)
    # Resize
    if bboxes is not None:
        bboxes = resizeBBoxes(bboxes,h/size)
    # Check Result
    if isdebug and bboxes is not None:
        drawBBox(origin,bboxes,debug=True)
    # Crop Rois
    rois = BBoxes2ROIs(origin,bboxes)
    if isdebug and rois is not None:
        for i in range(len(rois)):
            showResult("cropped",rois[i])
            
    return bboxes,rois
####################################################################################
################################Cascade-Based#######################################
####################################################################################
watch_cascade = cv2.CascadeClassifier('./model/license-plate.xml')

def computeSafeRegion(shape,bounding_rect):
    top = bounding_rect[1] # y
    bottom  = bounding_rect[1] + bounding_rect[3] # y +  h
    left = bounding_rect[0] # x
    right =   bounding_rect[0] + bounding_rect[2] # x +  w

    min_top = 0
    max_bottom = shape[0]
    min_left = 0
    max_right = shape[1]

    # print "computeSateRegion input shape",shape
    if top < min_top:
        top = min_top
        # print "tap top 0"
    if left < min_left:
        left = min_left
        # print "tap left 0"

    if bottom > max_bottom:
        bottom = max_bottom
        #print "tap max_bottom max"
    if right > max_right:
        right = max_right
        #print "tap max_right max"

    # print "corr",left,top,right,bottom
    return [left,top,right-left,bottom-top]


def cropped_from_image(image,rect):
    x, y, w, h = computeSafeRegion(image.shape,rect)
    return image[y:y+h,x:x+w]

# Image2BBoxes
def detect_by_cascade(origin,
                      resize_h=720,
                      en_scale =1.06,
                      isdebug=False):
    if origin is None:
        return None
    # Default Size
    height,width,channels = origin.shape
    # Resize
    resized = cv2.resize(origin, (int(width*resize_h/height), resize_h))    
    # Colr2Gray     
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    # Detect by Cascade
    watches = watch_cascade.detectMultiScale(gray, en_scale, 1, minSize=(36, 9))
    
    cropped_images = []
    bboxes = []
    for (x, y, w, h) in watches:
        xmin = x - w * 0.1
        ymin = y - h * 0.6
        xmin = xmin if xmin >= 0 else 0
        xmin = ymin if ymin >= 0 else 0
        xmax = xmin + 1.2 * w
        ymax = ymin + 1.1 * h
        bboxes.append(np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]))

        cropped = cropped_from_image(gray, (int(xmin), int(ymin), int(1.2*w), int(1.1*h)))
        cropped_images.append(cropped)
    # Resize
    if bboxes is not None:
        bboxes = resizeBBoxes(bboxes,height/float(resize_h))

     # Check Result
    if isdebug and bboxes is not None:
        drawBBox(origin,bboxes,debug=True)
    # Crop Rois
    rois = BBoxes2ROIs(origin,bboxes)
    if isdebug and rois is not None:
        for i in range(len(rois)):
            showResult("cropped",rois[i])
            
    return bboxes,rois#cropped_images

######################################################################################
######################################################################################
######################################################################################

##### Test Variable
dataset_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Tmp/LP/"
filename = "29.jpg"
fullpath = dataset_path + filename

import sys

def main(Mode="Test"):
    if Mode == "Test":
        img = cv2.imread(fullpath)
        detect_by_probability(img,isdebug=True)
    else:
        path =  sys.argv[1]
        img = cv2.imread(path)
        bboxes = detect_by_seg_gf(img)#DetectLP(fullpath)
        print(bboxes)    

if __name__ == "__main__":
    main()