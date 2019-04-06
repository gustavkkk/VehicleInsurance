#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 10:14:48 2017

@author: ubuntu
"""

import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from feature.bbox import drawBBox,analyzeBBox

import sys
sys.path.append('../')

from nets import ssd_vgg_300, np_methods#, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
#from notebooks import visualization

class VehicleDetector(object):
    def __init__(self, params=None):
        # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
        self.gpu_options = tf.GPUOptions(allow_growth=True)
        self.config = tf.ConfigProto(log_device_placement=False, gpu_options=self.gpu_options)
        # Input placeholder.
        self.net_shape = (300, 300)
        self.data_format = 'NHWC'
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        # Evaluation pre-processing: resize to SSD net shape.
        self.image_pre, self.labels_pre, self.bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            self.img_input, None, None, self.net_shape, self.data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        self.image_4d = tf.expand_dims(self.image_pre, 0)
        # Define the SSD model.
        self.reuse = True if 'ssd_net' in locals() else None
        self.ssd_net = ssd_vgg_300.SSDNet()
        with slim.arg_scope(self.ssd_net.arg_scope(data_format=self.data_format)):
            self.predictions, self.localisations, _, _ = self.ssd_net.net(self.image_4d, is_training=False, reuse=self.reuse)
        # Restore SSD model.
        self.ckpt_filename = './checkpoints/ssd_300_vgg.ckpt'
        # ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
        # SSD default anchor boxes.
        self.ssd_anchors = self.ssd_net.anchors(self.net_shape)

        self.isess = tf.InteractiveSession(config=self.config)     
        # Load Model     
        self.isess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.isess, self.ckpt_filename)
        
    def __enter__(self):
        return self
    
    # Main image processing routine.
    def process_image(self,
                      img,
                      select_threshold=0.5,
                      nms_threshold=.45,
                      net_shape=(300, 300)):
        # Run SSD network.
        rimg, rpredictions, rlocalisations, rbbox_img = self.isess.run([self.image_4d,self.predictions, self.localisations, self.bbox_img],
                                                                  feed_dict={self.img_input: img})
        
        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                rpredictions, rlocalisations, self.ssd_anchors,
                select_threshold=select_threshold, img_shape=self.net_shape, num_classes=21, decode=True)
        
        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes
    

    
    def detect(self,
               img,
               debug=False):
        
        rclasses, rscores, rbboxes =  self.process_image(img)
        #visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
        # Refine BBoxes
        bbox = VehicleDetector.pick_one_vehicle(img,rclasses, rscores, rbboxes)
        if debug:
            drawBBox(img,[bbox],drawmline=False)
        return bbox
                
    
    def detect_by_filename(self,
                           path,
                           debug=False):        
        # Load File
        img = mpimg.imread(path)#img = cv2.imread(path)
        # Detect
        bbox = self.detect(img)#mpimg.imread(path)
        # Check Result
        if bbox is not None and debug:
            drawBBox(img,[bbox])
        ##
        return bbox
        
    def detect_by_data(self,
                       img,
                       debug=False):         
        # Detect
        bbox = self.detect(img)#mpimg.imread(path)
        # Check Result
        if bbox is not None and debug:
            drawBBox(img,[bbox])
        ##
        return bbox

    @staticmethod    
    def rbbox2bbox(img,
                   rbbox):
        height,width,channels = img.shape
        ymin = int(rbbox[0] * height)
        xmin = int(rbbox[1] * width)
        ymax = int(rbbox[2] * height)
        xmax = int(rbbox[3] * width)
        return np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])
    
    @staticmethod
    def pick_one_vehicle(img,rclasses, rscores, rbboxes):
        size = 0
        selected = None
        for i,rclass in enumerate(rclasses):
            if rclass == 7 and rscores[i]>0.5:
                bbox = VehicleDetector.rbbox2bbox(img,rbboxes[i])
                pt0_,pt1_,pt2_,pt3_,w,h,center,pt_left,pt_right,angle_degree = analyzeBBox(bbox)
                if w*h > size:
                    selected = bbox
                    size = w*h
        return selected
    
    @staticmethod   
    def pick_vehicles(img,rclasses, rscores, rbboxes):
        bboxes = []
        sizes = []
        for i,rclass in enumerate(rclasses):
            if rclass == 7 and rscores[i]>0.5:
                bbox = VehicleDetector.rbbox2bbox(img,rbboxes[i])
                bboxes.append(rbboxes[i])
                pt0_,pt1_,pt2_,pt3_,w,h,center,pt_left,pt_right,angle_degree = analyzeBBox(bbox)
                sizes.append(w*h)
        return bboxes
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.isess.close()
        
from quality import AnalyzeImageQuality
from misc.switch import switch
from feature.extractfeature import refinedGoodFeatures,checkFeatures
from feature.threshold import compositeThreshold
from feature.bbox import findBBox,resizeBBoxes,resizeBBox,BBoxes2ROIs,showResult,close#,refineBBox
from feature.space import Laplacian,DoG,maskize,AdaptiveThreshold,tophatmask,NormalizeT#,NormalizeEx#,TopHat
from misc.preprocess import maximizeContrast as icontrast

import time

def refine_gfimage(img,mask):#time-consuming-mask
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
    #img_contours,contours,hierarchy = cv2.findContours(newmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours,hierarchy = cv2.findContours(newmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    img_contours = np.zeros((img.shape),dtype=np.uint8)
    if len(contours) == 0:
        return mask
    for contour in contours:
        random.shuffle(contour)
        cv2.drawContours(img_contours,[contour],0,(255,255,255),1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    newmask=cv2.morphologyEx(img_contours[:,:,0], cv2.MORPH_CLOSE, kernel)
    return newmask

def colormask(image,
              isday=True,
              tgtcolr='Default',
              isdebug=False):
    #
    #image = NormalizeT(image)
    #showResult("normalized",image)
    # blue mask + yellow mask
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    sat = (s.copy()).astype('float')
    val = (v.copy()).astype('float')
    sat /= 255
    val /= 255
    #
    hue = np.zeros(image.shape[:2])
    hue[hue>=0] = 0.2
    #
    blue1 = h.copy()
    blue2 = h.copy()
    yellow = h.copy()
    #
    if isday:
        blue1 = np.where((h<105)|(h>135)|(v<40)|(s<40),0,255)
        blue2 = np.where((h<100)|(h>140)|(v<30)|(s<30),0,255)
        yellow = np.where((h<20)|(h>40)|(v<40)|(s<40),0,255)
    else:
        blue1 = np.where((h<105)|(h>135),0,255)
        blue2 = np.where((h<100)|(h>140),0,255)
        yellow = np.where((h<20)|(h>40),0,255)
    #
    for case in switch(tgtcolr):
        if case('Blue'):
            hue[blue2>0] = 0.5
            hue[blue1>0] = 1.0
            hue[yellow>0] = 0.1
            break
        if case('Yellow'):
            hue[yellow>0] = 1.0
            hue[blue2>0] = 0.1
            hue[blue1>0] = 0.15
            break
        if case('Default'):
            hue[blue2>0] = 0.4
            hue[blue1>0] = 0.9
            hue[yellow>0] = 0.7
            break
    #
    if isday:
        mask=(hue*sat*val)**2#(hue*sat*val)**2
    else:
        mask=(hue*sat)**2
    #
    if isdebug:
        showResult("colormask:mask",mask)
    
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

colrs = {0:'Blue',1:'Yellow',2:'Default'}

def mkfinalmasks(image,
                gf_image=None,
                isday=True,
                isdebug=False):
    masks = []
    regmask = regionmask(image)
    for i in range(2):
        colrmask = colormask(image,isday,tgtcolr=colrs[i],isdebug=isdebug)
        mask = colrmask * regmask
        if gf_image is not None:
            gfmask = featuremask(gf_image)
            mask *= gfmask
        mask = (mask * 255).astype('uint8')
        if isdebug:
            showResult("mkfinalmask:mask",mask)
        masks.append(mask)
    return masks

def mask2plates(img,finalmask,isdebug=False):
    #showResult("masktest",finalmask)
    #ret,binary = cv2.threshold(finalmask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)\
    binary = compositeThreshold(finalmask,mode='otsu')
    closing=close(binary)
    if isdebug:
        showResult("mask2plates:masktest",closing)
    # Find Candidate
    return findBBox(img,closing,isdebug=isdebug)

from licenseplate import LicensePlate

class LicensePlateDetector:
    def __init__(self, image=None):
        if image is not None:
            self.bgr= image
        self.licenplatevalidator = LicensePlate()
        self.confidence = 0.0
        self.isFound = False
            
    def initialize(self):
        self.bgr = None

    def preprocess(self,image):
        self.bgr = image
        gray = cv2.cvtColor(self.bgr,cv2.COLOR_BGR2GRAY)
        self.gray = icontrast(255 - gray)
        self.laplacian = Laplacian(gray)
        #self.sobel = Sobel(gray)
        #self.entropy = Entropy(gray)
        #self.garbor = Garbor(gray)
        self.DoG = DoG(gray)
        self.lthr = AdaptiveThreshold(self.laplacian)
        self.tophat = tophatmask(gray)
        #tophatblackhat(gray)
        masks = []
        #masks.append(self.tophat)
        masks.append(self.DoG)
        self.compose = maskize(self.lthr,masks)
        self.contour = np.zeros((self.compose.shape[:2]),np.uint8)

    def showall(self):
        showResult("laplacian",self.laplacian)
        showResult("lthr",self.lthr)
        #showResult("DoG",self.DoG)
        showResult("tophat",self.tophat)
        showResult("compose",self.compose)
        
    def process(self,image):
        return self.detect(image)
        #self.preprocess(image)
        #self.showall()
      
    def detect(self,
               origin,
               isdebug=False):
        start = time.time()
        # Default Size
        h,w,c = origin.shape
        size = 200.0
        # Resize
        img = cv2.resize(origin,(int(w*size/h),int(size)))
        #showResult("img",img)
        for case in switch(AnalyzeImageQuality.dayornight(img)):
            if case('Day'):
                # Extract Good Features
                corners = refinedGoodFeatures(origin,img)
                mask = checkFeatures(img,corners,isdebug)
                closing=close(mask)
                refined_gfmask = refine_gfimage(img,closing)
                #showResult("refined_gfmask",refined_gfmask)
                finalmasks = mkfinalmasks(img,refined_gfmask,isday=True,isdebug=isdebug)
                break
            if case('Night'):
                finalmasks = mkfinalmasks(img,None,isday=isdebug)
                break

        for colrindex,fmask in enumerate(finalmasks):
            if (fmask>0).sum() == 0:
                continue
            bboxes = mask2plates(img,fmask)
            # Resize
            if bboxes is not None:
                bboxes = resizeBBoxes(bboxes,h/size)
                rois = BBoxes2ROIs(origin,bboxes)
                for i,roi in enumerate(rois):
                    confidence = self.licenplatevalidator.process(roi,mode=colrs[colrindex],isdebug=isdebug)
                    print confidence
                    if confidence > 0.7:
                        #pts = self.licenplatevalidator.getRefinedROI()
                        #bbox = refineBBox(bboxes[i],pts)
                        bbox = resizeBBox(bboxes[i],ratio=0.9)
                        print("total elapsed time: "+str(int((time.time() - start)*1000)/1000.0)+"s")
                        return confidence,[bbox],[roi]
        '''            
        # Check Result
        if isdebug and bboxes is not None:
            drawBBox(origin,bboxes,debug=True)
            for i in range(len(rois)):
                showResult("cropped",rois[i])
        '''        
        return 0.0,None,None

###
dataset_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Tmp/LP/"
filename = "29.jpg"
fullpath = dataset_path + filename
###

def TestVehicleDetector():
    vehicledetector = VehicleDetector()
    vehicledetector.detect_by_filename(fullpath)
    
def TestLicensePlateDetector():
    licensedetector = LicensePlateDetector()
    licensedetector.process(cv2.imread(fullpath))

def main():
    TestLicensePlateDetector()
    
if __name__ == '__main__':
    print("Under Test")
    main()
    
