#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 22:49:10 2017

@author: ubuntu

ref:https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
"""
#from imutils import paths
#import argparse
import cv2

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

class AnalyzeImageQuality():
    def __init__(self,image=None):
        if image is not None:
            self.image = image
            self.preprocess()
        self.result = True
        self.explanation = ""
        
    def preprocess(self):
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.fm = variance_of_laplacian(self.gray)
     
    def isBlurred(self,threshold=100.0):
        if 100 < self.fm and self.fm < 400:
            self.result = False
            self.explanation = "reflective"
        elif 40 <= self.fm and self.fm <= 100:
            self.result = False
            self.explanation = "blurry"            
        elif self.fm < 40:
            self.result = False
            self.explanation = "Too reflective"
        else:
            self.result = True
            self.explanation = "Normal"
        return self.fm,self.result,self.explanation
##### Test Variable
dataset_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Tmp/VIN/"
problem_db_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Focus-VIN/"
filename = "42.jpg"
fullpath = dataset_path + filename
     
if __name__ == "__main__":
    image = cv2.imread(fullpath)
    aiq = AnalyzeImageQuality(image)
    print aiq.isBlurred()