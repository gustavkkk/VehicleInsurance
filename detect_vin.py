#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:39:32 2017

@author: ubuntu

ref:https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
"""
import matplotlib.pyplot as plt
import numpy as np

import cv2

from feature.denoise import Denoise
#from feature.colorspace import rgb2hsv,opencv2skimage,skimage2opencv,checkBlue,checkYellow,equalizehist
from feature.extractfeature import refinedGoodFeatures,checkFeatures
from feature.bbox import findBBox,drawBBox,resizeBBoxes,showResult
#import time

import math
import random

from misc.contour import Contour,ROI
from misc.preprocess import preprocess,ADAPTIVE_THRESH_BLOCK_SIZE,ADAPTIVE_THRESH_WEIGHT
#from misc.switch import switch
# Colr Definition
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

def find_possible_chars(thr,
                        isdebug=False):
    #
    tmp = thr.copy()
    img_contour, contours, npaHierarchy = cv2.findContours(tmp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # find all contours
    height, width = thr.shape
    img_contour = np.zeros((height, width, 3), np.uint8)
    #
    filtered = []                # this will be the return value
    count = 0
    for i in range(0, len(contours)):                       # for each contour

        if isdebug:
            cv2.drawContours(img_contour, contours, i, SCALAR_WHITE)
            
        contour = Contour(contours[i])

        if contour.checkIfPossibleChar():                   # if contour is a possible char, note this does not compare to other chars (yet) . . .
            count += 1                                      # increment count of possible chars
            filtered.append(contour)                        # and add to list of possible chars

    if isdebug:
        print "\nstep 2 - contours = " + str(len(contours))       # 2362 with MCLRNF1 image
        print "step 2 - chars = " + str(count)       # 131 with MCLRNF1 image
        showResult("contours",img_contour)
    return filtered
  
# module level variables ##########################################################################
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5

###################################################################################################
def extractROI(imgOriginal, string):
    roi = ROI()           # this will be the return value

    string.sort(key = lambda char: char.brcX)        # sort chars from left to right based on x position

    # calculate the center point of the plate
    center_x = (string[0].brcX + string[len(string) - 1].brcX) / 2.0
    center_y = (string[0].brcY + string[len(string) - 1].brcY) / 2.0
    center = center_x, center_y

    # calculate plate width and height
    intPlateWidth = int((string[len(string) - 1].brX + string[len(string) - 1].brW - string[0].brX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in string:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.brH
    # end for

    fltAverageCharHeight = intTotalOfCharHeights / len(string)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    # calculate correction angle of plate region
    fltOpposite = string[len(string) - 1].brY - string[0].brY
    fltHypotenuse = Contour.distanceBetweenChars(string[0], string[len(string) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    roi.rrLocationOfPlateInScene = (tuple(center), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

    # final steps are to perform the actual rotation

    # get the rotation matrix for our calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(center), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape      # unpack original image width and height

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))       # rotate the entire image

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(center))

    roi.imgPlate = imgCropped         # copy the cropped plate image into the applicable member variable of the possible plate

    return roi

def drawChars(img,
            title,
            contours,
            colr=SCALAR_WHITE,
            isdebug=False):
    img_contours = np.zeros((img.shape), np.uint8)

    contours_ = []
    for contour in contours:
        contours_.append(contour.contour)
        
    cv2.drawContours(img_contours, contours_, -1, colr)
    if isdebug:
        showResult(title,img_contours)
    return img_contours

def contour2area(contour):
    #contour = numpy.array([[[0,0]], [[10,0]], [[10,10]], [[5,4]]])
    return cv2.contourArea(contour)

def warp(img,
         pts1=np.float32([[56,65],[368,52],[28,387],[389,390]]),
         pts2=np.float32([[0,0],[300,0],[0,300],[300,300]])):
    rows,cols,ch = img.shape
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(300,300))
    #plt.subplot(121),plt.imshow(img),plt.title('Input')
    #plt.subplot(122),plt.imshow(dst),plt.title('Output')
    #plt.show()    
    return dst

def detect_by_contour(origin,
                      isdebug=False):
    # Initialize
    height, width, numChannels = origin.shape
    img_gray = np.zeros((height, width, 1), np.uint8)
    img_thr = np.zeros((height, width, 1), np.uint8)
    img_contour = np.zeros((height, width, 3), np.uint8)
    # Grayscale
    img_gray, img_thr = preprocess(origin)
    if isdebug:
        showResult("img_gray",img_gray)
        showResult("img_thr",img_thr)
        #showResult("Test",cv2.Canny(img_gray,50,200))
    # First Filtering(Contours2Chars)
    chars = find_possible_chars(img_thr)
    if isdebug:
        print "step 2 - the numbder of suspicious chars(roughly filtered contours) = " + str(len(chars))
        drawChars(img_contour,"first-filtering",chars,isdebug=True)
    # Second Filtering(Chars2Strings)
    strings = Contour.findListOfListsOfMatchingChars(chars)
    if isdebug: # show steps #######################################################
        print "step 3 - strings.Count = " + str(len(strings))    # 13 with MCLRNF1 image
        img_contour = np.zeros((height, width, 3), np.uint8)

        for string in strings:
            (b,g,r) = (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
            img_contour = drawChars(img_contour,"second-filtering",string,(b,g,r))
    # Third Filtering(String2ROIs)
    ROIs = []
    bboxes = []
    for string in strings:                   # for each group of matching chars
        roi = extractROI(origin, string)         # attempt to extract plate

        if roi.imgPlate is not None:                          # if plate was found
            ROIs.append(roi)                  # add to list of possible plates
            bbox = cv2.boxPoints(roi.rrLocationOfPlateInScene)
            ibbox = bbox.astype(int)
            bboxes.append(ibbox)
            
    # 
    print "\n" + str(len(ROIs)) + " possible plates found"          # 13 with MCLRNF1 image

    if isdebug and len(ROIs) != 0: # show steps #######################################################

        for i in range(len(ROIs)):
            p2fRectPoints = cv2.boxPoints(ROIs[i].rrLocationOfPlateInScene)

            cv2.line(origin, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
            cv2.line(origin, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
            cv2.line(origin, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
            cv2.line(origin, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
            showResult("roi",ROIs[i].imgPlate)

        showResult("result",origin)

    return bboxes
# end function
    
import sys

##### Test Variable
dataset_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Tmp/VIN/"
filename = "11.jpg"
fullpath = dataset_path + filename

def main(Mode="Test"):
    if Mode == "Test":
        img = cv2.imread(fullpath)
        detect_by_contour(img,True)
    else:
        path =  sys.argv[1]
        img = cv2.imread(path)
        bboxes = detect_by_contour(img)#DetectLP(fullpath)
        print(bboxes)    

if __name__ == "__main__":
    main()