#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 22:49:10 2017

@author: junying

ref:https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
"""
#from imutils import paths
#import argparse
import cv2
import numpy as np

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

class AnalyzeImageQuality():
    def __init__(self,image=None):
        self.initialize()
        if image is not None:
            self.image = image

    def initialize(self):
        self.image, self.gray = None, None
        self.fm = 0.0
        self.result = True
        self.explanation = ""        
        
    def preprocess(self,image):
        self.image = image
        h,w,c = image.shape
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.pblur = variance_of_laplacian(self.gray)
        self.pbright = np.sum(self.gray) / (h*w)

    @staticmethod
    def dayornight(img):
        h,w,c = img.shape
        avg = np.sum(img) / (w*h*c)
        print(avg)
        if avg > 50:#70:
            return 'Day'
        else:
            return 'Night'
     
    def analyze(self,threshold=100.0):
        if 100 < self.pblur and self.pblur < 400:
            self.result = False
            self.explanation = "Reflective"
        elif 40 <= self.pblur and self.pblur <= 100:
            self.result = False
            self.explanation = "Blurry"            
        elif self.pblur < 40:
            self.result = False
            self.explanation = "Too Reflective"
        else:
            self.result = True
            self.explanation = "Normal"
        return self.pblur,self.result,self.explanation
    
    def process(self,image):
        self.initialize()
        self.preprocess(image)
        return self.analyze()
##### Test Variable
dataset_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Tmp/LP/"
problem_db_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Focus-VIN/"
filename = "29.jpg"
fullpath = dataset_path + filename
     
if __name__ == "__main__":
    import time
    s = time.time()
    image = cv2.imread(fullpath)
    aiq = AnalyzeImageQuality(image)
    print(aiq.process(image))
    print(AnalyzeImageQuality.dayornight(image))
    print("elapsed time : " + str(int((time.time() - s)*1000)/1000.0) + "s ")