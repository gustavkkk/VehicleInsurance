#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 22:10:46 2017

@author: ubuntu

ref:http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html?highlight=rectangle
    https://stackoverflow.com/questions/42235429/python-opencv-shape-detection
    https://stackoverflow.com/questions/41879315/opencv-using-cv2-approxpolydp-correctly
"""
import cv2
from skimage import io
import numpy as np
from denoise import Denoise
#from extractfeature import morphological
from matplotlib import pyplot as plt
from feature.colorspace import checkBlue,checkYellow,rgb2hsv,opencv2skimage,skimage2opencv
from misc.switch import switch

def showResult(name,
               img):
    cv2.imshow(name,img)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
        
def contour2bbox(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box

def contour2fitline(contour):
    bbox =  contour2bbox(contour)
    pt0_,pt1_,pt2_,pt3_,w,h,center,pt_left,pt_right,angle_degree = analyzeBBox(bbox)
    [vx, vy, x, y] = cv2.fitLine(np.array([pt_left,pt_right]), cv2.DIST_HUBER, 0, 0.01, 0.01)
    return [vx, vy, x, y],bbox
    
def contour2convex(contour):
    hull = cv2.convexHull(contour,returnPoints = False)
    defects = cv2.convexityDefects(contour,hull)
    pts = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(contour[s][0])
        #end = tuple(contour[e][0])
        #far = tuple(contour[f][0])
        pts.append(start)
    return pts
# lp position = ratio * lp position
def resizeBBoxes(bboxes,
                 ratio):
    #resize1
    bboxes_ = []
    for bbox in bboxes:
        bbox = resizeBBox(bbox,ratio=1.3)
        bboxes_.append(bbox)
    #resize2
    bboxes = np.array(bboxes_,dtype=np.float64)#,dtype=np.float64)
    bboxes *= ratio
    bboxes = np.int_(bboxes)
    bboxes = list(bboxes)#.tolist()
    return bboxes

# lp position = car position + lp position
def shiftBBoxes(bbox_car,bboxes_lp):
    if bboxes_lp is None or bbox_car is None:
        return None
    xmin = bbox_car[0][0]
    ymin = bbox_car[0][1]
    refined = []
    for bbox in bboxes_lp:
        for index in range(bbox.shape[0]):
            bbox[index,0] += xmin
            bbox[index,1] += ymin
        refined.append(bbox)
    return np.array(refined)

def estimateColr(img,
                 rect,
                 debug=False):
    [x, y, w, h] = rect
    roi = img[y:y+h, x:x+w]
    rgb = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
    totalcolr = [0,0,0]
    for j in range(h):
        for i in range(w):
            totalcolr += rgb[j,i]
    pixelcount = w * h
    avergecolr = totalcolr / pixelcount
    if debug:
        print("estimatecolr,avrgcolr:",avergecolr)
    hsv = rgb2hsv(avergecolr)
    return checkBlue(hsv) or checkYellow(hsv)

def analyzeBBox(bbox):
    pt1 = (int((bbox[0][0] + bbox[3][0]) / 2),int((bbox[0][1] + bbox[3][1]) / 2))
    pt2 = (int((bbox[1][0] + bbox[2][0]) / 2),int((bbox[1][1] + bbox[2][1]) / 2))
    pt3 = (int((bbox[0][0] + bbox[1][0]) / 2),int((bbox[0][1] + bbox[1][1]) / 2))
    pt4 = (int((bbox[2][0] + bbox[3][0]) / 2),int((bbox[2][1] + bbox[3][1]) / 2))
    center= (int((pt1[0]+pt2[0])/2),int((pt1[1]+pt2[1])/2))
    w = np.linalg.norm(np.array(pt1)-np.array(pt2))
    h = np.linalg.norm(np.array(pt3)-np.array(pt4))
    ###
    if w < h:
        tmp = w
        w = h
        h = tmp
        if pt3[0] < pt4[0]:
            pt_left = pt3
            pt_right = pt4
            [pt0_,pt3_] = [bbox[0],bbox[1]] if bbox[0][1] < bbox[1][1] else [bbox[1],bbox[0]]
            [pt1_,pt2_] = [bbox[2],bbox[3]] if bbox[2][1] < bbox[3][1] else [bbox[3],bbox[2]]
        else:
            pt_left = pt4
            pt_right = pt3
            [pt0_,pt3_] = [bbox[2],bbox[3]] if bbox[2][1] < bbox[3][1] else [bbox[3],bbox[2]]
            [pt1_,pt2_] = [bbox[0],bbox[1]] if bbox[0][1] < bbox[1][1] else [bbox[1],bbox[0]]
    else:
        if pt1[0] < pt2[0]:
            pt_left = pt1
            pt_right = pt2
            [pt0_,pt3_] = [bbox[0],bbox[3]] if bbox[0][1] < bbox[3][1] else [bbox[3],bbox[0]]
            [pt1_,pt2_] = [bbox[1],bbox[2]] if bbox[1][1] < bbox[2][1] else [bbox[2],bbox[1]]
        else:
            pt_left = pt2
            pt_right = pt1
            [pt0_,pt3_] = [bbox[1],bbox[2]] if bbox[1][1] < bbox[2][1] else [bbox[2],bbox[1]]
            [pt1_,pt2_] = [bbox[0],bbox[3]] if bbox[0][1] < bbox[3][1] else [bbox[3],bbox[0]]
            
    angle_radian = math.atan(float(pt_right[1] - pt_left[1])/w)
    angle_degree = angle_radian * (180.0 / math.pi)            
    
    return pt0_,pt1_,pt2_,pt3_,w,h,center,pt_left,pt_right,angle_degree

def cropImg_by_BBox(img,
                    bbox):
    tmp = img.copy()
    roi = tmp[bbox[0][1]:bbox[3][1],bbox[0][0]:bbox[1][0]]
    return roi

import math

def refineBBox(bbox,innerbbox):
    pt0_,pt1_,pt2_,pt3_,w,h,center,pt_left,pt_right,angle=analyzeBBox(bbox)
    [x0y0,x1y0,x1y1,x0y1] = innerbbox
    ordered = [pt0_,pt1_,pt2_,pt3_]
    bbox_ = []
    #print "refineBBox:innerbbox",innerbbox
    pt = ordered[0]
    angle *= -1
    for i in range(4):
        pt_ = innerbbox[i]
        dist = np.linalg.norm(np.array(pt_))
        radian_ = math.acos(float(pt_[0])/dist)
        angle_ = radian_ * (180.0 / math.pi)
        delta_angle = angle_ - angle
            
        delta_x = dist*math.cos(delta_angle*math.pi/180)
        delta_y = dist*math.sin(delta_angle*math.pi/180)
        bbox_.append([pt[0]+int(delta_x),pt[1]+int(delta_y)])
        
    return np.array(bbox_)

def cropImg_by_BBox2(img,
                    bbox,
                    isdebug=False):
    tmp = img.copy()
    roi = tmp[bbox[0][1]:bbox[3][1],bbox[0][0]:bbox[1][0]]
    pt0_,pt1_,pt2_,pt3_,w,h,center,pt_left,pt_right,angle_degree=analyzeBBox(bbox)
    
    if isdebug:
        print("cropImg_by_BBox2:",pt_left,pt_right)
        img = cv2.line(img,pt_left,pt_right,(0,244,222),2)
        showResult("cropImg_by_BBox2-debug",img)

    # get the rotation matrix for our calculated correction angle
    rmatrix = cv2.getRotationMatrix2D(tuple(center), angle_degree, 1.0)
    height, width, numChannels = img.shape      # unpack original image width and height
    rotated = cv2.warpAffine(img, rmatrix, (width, height))       # rotate the entire image
    # Resize Cropping
    #w *= 1.3
    #h *= 1.3
    # Crop
    roi = cv2.getRectSubPix(rotated, (int(w), int(h)), tuple(center))

    return roi

def BBoxes2ROIs(img,bboxes):
    if bboxes is None:
        return None
    rois = []
    for bbox in bboxes:
        rois.append(cropImg_by_BBox2(img,bbox))
    return rois

def ContourFiltering(mask,
                     contours,
                     debug=False):
    height,width = mask.shape
    bboxes = []
    for contour in contours:
        #perimeter = cv2.arcLength(contour,True)
        #[x,y,ww,hh] = cv2.boundingRect(contour)
        #aspect_ratio = float(ww)/hh
        epsilon = 0.04*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,epsilon,True)
        bbox = contour2bbox(contour)
        w = np.linalg.norm(bbox[0]-bbox[1])
        h = np.linalg.norm(bbox[1]-bbox[2])
        w_by_h = w / h if h != 0 else w
        aspect_ratio = w_by_h if w_by_h < 1 else 1/w_by_h
        roi_by_origin = float(w * h) / (height*width)
        if debug:
            print("ContourFiltering,approx,aspect_ratio,roi_by_origin:\n",
                  approx,
                  aspect_ratio,
                  roi_by_origin)

        if len(approx) > 3 and len(approx) < 7\
            and 0.0022 < roi_by_origin and roi_by_origin < 0.15\
            and 0.2 < aspect_ratio and aspect_ratio < 0.75:
            #estimateColr(img,boundingRect):
            bboxes.append(bbox)#contour2convex(contour))#bboxes.append(bbox)
            
    return bboxes

def removeBoundary(mask):
    h,w = mask.shape
    mask[0:int(0.25*h),:] = 0
    mask[int(0.98*h):h-1,:] = 0
    mask[:,0:int(0.02*w)] = 0
    mask[:,int(0.98*w):w-1] = 0
     
def split(image,
          isdebug=False):
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    h,s,v = cv2.split(hsv)
    #
    removeBoundary(h)
    #
    blue = h.copy()
    yellow = h.copy()
    blue = np.where((h<105)|(h>135)|(v<100),0,255)
    yellow = np.where((h<20)|(h>40),0,255)
    if isdebug:
        showResult("blue",blue)
        showResult("yellow",yellow)
    
    imgs = []
    imgs.append(blue)
    imgs.append(yellow)
    return imgs

def close(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

def mask2contour(mask,
                 isdebug=False):
    #
    closing=close(mask)
    # Find Contours
    #imgContours,contours,hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
    contours,hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Filter Contours
    bboxes = ContourFiltering(mask,contours,isdebug)
    
    return bboxes

def grabcut(origin,
            handwrite):
    #
    h,w,c = origin.shape
    img = origin.copy()
    mask = np.zeros(img.shape[:2],np.uint8)
    mask[mask>=0]=2
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    #rect = (50,50,w-50,h-50)
    #cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)    
    #mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    #img = img*mask2[:,:,np.newaxis]
    #
    mask[handwrite == 0] = 0
    mask[handwrite == 255] = 1    
    mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)    
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask[:,:,np.newaxis]
    showResult("grabcut",img)
    return img

from skimage import morphology
from skimage.segmentation import find_boundaries
from skimage import img_as_ubyte

# maintain the center of bbox
def resizeBBox(bbox,
               ratio = 1.1):
    #
    x_c,y_c=(0,0)
    for index in range(bbox.shape[0]):
        x_c += bbox[index][0]
        y_c += bbox[index][1]
    x_c /= 4
    y_c /= 4
    radius = math.sqrt((bbox[0][0] - x_c)**2 +(bbox[0][1] - y_c)**2)
    #
    bbox_resized = bbox.copy()
    for index in range(bbox.shape[0]):
        y_by_r = (float(bbox[index][1] - y_c)/radius)
        x_by_r = (float(bbox[index][0] - x_c)/radius)
        bbox_resized[index][1] = y_by_r * radius * ratio + y_c
        bbox_resized[index][0] = x_by_r * radius * ratio + x_c
        # exception handling            
    return bbox_resized

#
def mimicHandwrite(mask,
                   isdebug=True):
    
    #Initializa Handwrite
    handwrite = np.zeros((mask.shape),dtype=np.uint8)
    handwrite = 100 - handwrite
    #fg mark
    binary = mask.copy()
    binary[binary>0]=1
    skeleton = img_as_ubyte(morphology.skeletonize_3d(binary))#skeletonize(binary))
    skeleton[skeleton > 0] = 255
    handwrite[mask>0] = 255
    #bg mark
    bboxes = mask2contour(mask)
    if len(bboxes) != 0:
        bbox_resized = resizeBBox(bboxes[0])
        bg_colr = np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.uint8)
        cv2.drawContours(bg_colr,[bbox_resized],-1,(255,255,255),2)
        bg_mask = bg_colr[:,:,1]
        #dilate[dilate>0] = 1
        #boundary = find_boundaries(dilate, mode='inner').astype(np.uint8)
        handwrite[bg_mask>0] = 0
    else:
        return None
    
    if isdebug:
        showResult("handwrite",handwrite)
        
    return handwrite

# image = mask*image
def masked(img,
           mask,
           isdebug=False):
    if img.shape != mask.shape:
        mask = cv2.resize(mask,(img.shape[1],img.shape[0]))
    print mask.shape
    mask[mask>0]=1
    masked = img.copy()
    masked = masked*mask[:,:,np.newaxis]
    if isdebug:
        showResult("masked",masked)
    return masked

def findBBox(img,
             colr,
             isdebug=False):
    if len(colr.shape) == 3:
        # colr mask
        masks = split(colr)
        bboxes = []
        for mask in masks:
            if isdebug:
                showResult("findBBox:mask",mask)        
            bboxes += mask2contour(mask)
    else:
        mask = colr
        removeBoundary(mask)
        if isdebug:
            showResult("findBBox:mask",mask)
        bboxes = mask2contour(mask,isdebug)
        
    if len(bboxes) == 0:
        return None
    
    return bboxes

def drawBBox(img,
            bboxes_lp,
            bbox_car=None,
            bxcolor1 = (255,255,0),
            bxcolor2 = (0,255,255),
            linecolor = (0,255,0),
            drawmline=False,
            debug=False):
    
    out = img.copy()
    
    if bbox_car is not None:
        cv2.drawContours(out,[bbox_car],-1,bxcolor1,2)
        
    if bboxes_lp is None:
        return out
    
    for bbox in bboxes_lp:
        cv2.drawContours(out,[bbox],-1,bxcolor2,2)     
        # Direction Line ?x,y?y,x
        if drawmline:
            pt0_,pt1_,pt2_,pt3_,w,h,center,pt_left,pt_right,angle_degree = analyzeBBox(bbox)
            out = cv2.line(out,pt_left,pt_right,linecolor,2)
    if debug:
        showResult("drawBBox:result",out)
    return out
