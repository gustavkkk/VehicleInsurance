#coding=utf-8
import  finemapping  as  fm

import segmentation

import cv2

import time

from detect_lp import detect_by_seg_gf,detect_by_probability

from feature.bbox import showResult
from misc.preprocess import maximizeContrast
from feature.threshold import kojy_gray

def recognizeLP(gray):
    t0 = time.time()
    if gray is None:
        return

    gray = cv2.resize(gray,(136,36))

    blocks,res,confidence = segmentation.slidingWindowsEval(gray)
    for i in range(len(blocks)):
        showResult("plate-gray,",blocks[i])
    if confidence>4.5:
        print "车牌:",res,"置信度:",confidence
    else:
        print "不确定的车牌:", res, "置信度:", confidence

    print time.time() - t0,"s"
    
    return blocks,res,confidence

def recognizeLP2(seg_blocks,mid):
    t0 = time.time()
    blocks,res,confidence = segmentation.recogChars(seg_blocks,mid)
    if confidence>4.5:
        print "车牌:",res,"置信度:",confidence
    else:
        print "不确定的车牌:", res, "置信度:", confidence

    print time.time() - t0,"s"
    
    return blocks,res,confidence
    
def SimpleRecognizePlate(image):
    t0 = time.time()
    bboxes,images = detect_by_probability(image)#detect.detectPlateRough(image)
    if images is None:
        return
    for image in images:
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image,(136,36))
        image = kojy_gray(image)
        #image = maximizeContrast(255 - image)

        image_gray = fm.findContoursAndDrawBoundingBox(image)
        showResult("plate-colr",image)
        showResult("plate-gray,",image_gray)

        blocks,res,confidence = segmentation.slidingWindowsEval(image_gray)
        #for i in range(len(blocks)):
        #    showResult("plate-gray,",blocks[i])
        if confidence>4.5:
            print "车牌:",res,"置信度:",confidence
        else:
            print "不确定的车牌:", res, "置信度:", confidence

    print time.time() - t0,"s"




