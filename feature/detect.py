#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 07:11:36 2017

@author: junying

https://stackoverflow.com/questions/4706118/read-numbers-and-letters-from-an-image-using-opencv
https://github.com/opencv/opencv_contrib/tree/master/modules/text/samples
https://github.com/opencv/opencv_contrib/tree/master/modules/text
"""
import cv2
import numpy as np

def showResult(name,
               img):
    cv2.imshow(name,img)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()

def textdetector(gray,
                 isdebug=False):  
    showResult("gray",gray)
    # found suspicious regions
    mser = cv2.MSER_create()#2,60,1200)
    regions = mser.detectRegions(gray)
    hulls = [cv2.convexHull(np.array(p).reshape(-1, 1, 2)) for p in regions[0]]
    # filtering regions by size
    h,w = gray.shape
    hulls_ = []
    for hull in hulls:
        boundingrect = cv2.boundingRect(hull)
        [x,y,ww,hh] = boundingrect
        if ww > w/6 or hh < h/2:
            continue
        hulls_.append(hull)

    if isdebug:
        vis = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, hulls_, 1, (0, 255, 0))
        #filtering
        '''
        rects = regions[1]
        #print rects
        for rect in rects:
            x,y,w,h = rect
            print rect
            print x,y,w,h
            #cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),-1)
            cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),1,8,0)
        '''
        showResult("vis",vis)
    # create text region mask
    mask = np.zeros((h,w,3), dtype=np.uint8)
    mask[mask>=0]=255
    for contour in hulls_:
        cv2.drawContours(mask, [contour], -1, (0, 0, 0), -1)
    
    showResult("textmask",mask)
    return mask[:,:,1]

#import os
#import sys

def textdetector2(img):
    print('\ntextdetection.py')
    print('       A demo script of the Extremal Region Filter algorithm described in:')
    print('       Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012\n')
    
    '''   
    if (len(sys.argv) < 2):
      print(' (ERROR) You must call this script with an argument (path_to_image_to_be_processed)\n')
      quit()
    
    pathname = os.path.dirname(sys.argv[0])
    
    
    img      = cv2.imread(str(sys.argv[1]))
    '''
    # for visualization
    vis      = img.copy()
    
    
    # Extract channels to be processed individually
    channels = cv2.text.computeNMChannels(img)
    # Append negative channels to detect ER- (bright regions over dark background)
    cn = len(channels)-1
    for c in range(0,cn):
      channels.append((255-channels[c]))
    
    # Apply the default cascade classifier to each independent channel (could be done in parallel)
    print("Extracting Class Specific Extremal Regions from "+str(len(channels))+" channels ...")
    print("    (...) this may take a while (...)")
    for channel in channels:
    
      erc1 = cv2.text.loadClassifierNM1('./trained_classifierNM1.xml')
      er1 = cv2.text.createERFilterNM1(erc1,16,0.00015,0.13,0.2,True,0.1)
    
      erc2 = cv2.text.loadClassifierNM2('./trained_classifierNM2.xml')
      er2 = cv2.text.createERFilterNM2(erc2,0.5)
    
      regions = cv2.text.detectRegions(channel,er1,er2)
    
      rects = cv2.text.erGrouping(img,channel,[r.tolist() for r in regions])
      #rects = cv2.text.erGrouping(img,channel,[x.tolist() for x in regions], cv2.text.ERGROUPING_ORIENTATION_ANY,'../../GSoC2014/opencv_contrib/modules/text/samples/trained_classifier_erGrouping.xml',0.5)
    
      #Visualization
      for r in range(0,np.shape(rects)[0]):
        rect = rects[r]
        cv2.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 0), 2)
        cv2.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)
        
    showResult("Text detection result", vis)
    
sample_db_path = "./sample/"
test_db_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Tmp/VIN/"
filename = "1.jpg"
fullpath = test_db_path + filename
if __name__ == "__main__":
    gray = cv2.imread(fullpath,0)#cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    textdetector(gray,True)
    #textdetector2(cv2.imread(fullpath))
    