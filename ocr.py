#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:32:26 2017

@author: junying
sudo apt install tesseract-ocr-all
"""

import sys
import cv2

##################################################################################################
##################################################################################################
##################################################################################################
import numpy as np
import tesserocr
from PIL import Image
from tesserocr import PyTessBaseAPI, RIL, PSM, OEM, iterate_level
from feature.bbox import showResult

def opencv2pillow(image):
    return Image.fromarray(image)

class TesserOCR(object):

    def __init__(self,image=None,lang='chi_sim',psm=PSM.SINGLE_CHAR):
        self.api = PyTessBaseAPI(lang=lang,psm=psm)

    @staticmethod    
    def getinfo():
        print(tesserocr.tesseract_version())  # print tesseract-ocr version
        print(tesserocr.get_languages())

    @staticmethod        
    def img2string(image):
        pimg = opencv2pillow(image)
        print(tesserocr.image_to_text(pimg))

    @staticmethod
    def analyzeRIL(image):
        pimg = opencv2pillow(image)
        with PyTessBaseAPI() as api:
            api.SetImage(pimg)
            boxes = api.GetComponentImages(RIL.TEXTLINE, True)
            print('Found {} textline image components.'.format(len(boxes)))
            for i, (im, box, _, _) in enumerate(boxes):
                # im is a PIL image object
                # box is a dict with x, y, w and h keys
                api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
                ocrResult = api.GetUTF8Text()
                conf = api.MeanTextConf()
                print (u"Box[{0}]: x={x}, y={y}, w={w}, h={h}, "
                       "confidence: {1}, text: {2}").format(i, conf, ocrResult, **box)
        
    def img2string_single(self,image):
        pimg = opencv2pillow(image)
        self.api.SetImage(pimg)
        return self.api.GetUTF8Text()

    @staticmethod
    def detect(image):
        pimg = opencv2pillow(image)
        with PyTessBaseAPI(psm=PSM.AUTO_OSD) as api:
            api.SetImage(pimg)
            api.Recognize()   
            it = api.AnalyseLayout()
            orientation, direction, order, deskew_angle = it.Orientation()
            print("Orientation: {:d}".format(orientation))
            print("WritingDirection: {:d}".format(direction))
            print("TextlineOrder: {:d}".format(order))
            print("Deskew angle: {:.4f}".format(deskew_angle))

    @staticmethod    
    def segmentation(image):
        pimg = opencv2pillow(image)
        with PyTessBaseAPI(psm=PSM.OSD_ONLY) as api:
            api.SetImage(pimg)    
            os = api.DetectOS()
            print ("Orientation: {orientation}\nOrientation confidence: {oconfidence}\n"
                   "Script: {script}\nScript confidence: {sconfidence}").format(**os)
            
    # this functions works with tesseract4+
    @staticmethod
    def analyzeLSTM(image):
        pimg = opencv2pillow(image)
        with PyTessBaseAPI(psm=PSM.OSD_ONLY, oem=OEM.LSTM_ONLY) as api:
            api.SetImage(pimg)
            os = api.DetectOrientationScript()
            print ("Orientation: {orient_deg}\nOrientation confidence: {orient_conf}\n"
                   "Script: {script_name}\nScript confidence: {script_conf}").format(**os)
    
    @staticmethod
    def rect2string(image,rect):
        with PyTessBaseAPI() as api:
            api.SetImage(image)
            api.SetVariable("save_blob_choices", "T")
            api.SetRectangle(rect)#(37, 228, 548, 31)
            api.Recognize()
        
            ri = api.GetIterator()
            level = RIL.SYMBOL
            for r in iterate_level(ri, level):
                symbol = r.GetUTF8Text(level)  # r == ri
                conf = r.Confidence(level)
                if symbol:
                    print(u'symbol {}, conf: {}'.format(symbol, conf))
                indent = False
                ci = r.GetChoiceIterator()
                for c in ci:
                    if indent:
                        print('\t\t ')
                    print('\t- ')
                    choice = c.GetUTF8Text()  # c == ci
                    print(u'{} conf: {}'.format(choice, c.Confidence()))
                    indent = True
                print('---------------------------------------------')
##################################################################################################
##################################################################################################
##################################################################################################
from subprocess import Popen, PIPE
import os
import tempfile

PROG_NAME = 'tesseract'
TEMP_IMAGE = tempfile.mktemp()+'.bmp'
TEMP_FILE = tempfile.mktemp()

#All the PSM arguments as a variable name (avoid having to know them)
PSM_OSD_ONLY = 0
PSM_SEG_AND_OSD = 1
PSM_SEG_ONLY = 2
PSM_AUTO = 3
PSM_SINGLE_COLUMN = 4
PSM_VERTICAL_ALIGN = 5
PSM_UNIFORM_BLOCK = 6
PSM_SINGLE_LINE = 7
PSM_SINGLE_WORD = 8
PSM_SINGLE_WORD_CIRCLE = 9
PSM_SINGLE_CHAR = 10

class TesseractException(Exception): #Raised when tesseract does not return 0
    pass

class TesseractNotFound(Exception): #When tesseract is not found in the path
    pass

class PyTesser(object):
    
    def __init__(self,image=None):
        self.image = image
    
    @staticmethod     
    def check_path(): #Check if tesseract is in the path raise TesseractNotFound otherwise
        for path in os.environ.get('PATH', '').split(':'):
            filepath = os.path.join(path, PROG_NAME)
            if os.path.exists(filepath) and not os.path.isdir(filepath):
                return True
        raise TesseractNotFound()

    @staticmethod         
    def process_request(input_file, output_file, lang=None, psm=None):
        args = [PROG_NAME, input_file, output_file] #Create the arguments
        if lang is not None:
            args.append("-l")
            args.append(lang)
        if psm is not None:
            args.append("-psm")
            args.append(str(psm))
        proc = Popen(args, stdout=PIPE, stderr=PIPE) #Open process
        ret = proc.communicate() #Launch it
    
        code = proc.returncode
        if code != 0:
            if code == 2:
                raise TesseractException("File not found")
            if code == -11:
                raise TesseractException("Language code invalid: "+ret[1])
            else:
                raise TesseractException(ret[1])   
                
    @staticmethod     
    def tostring(gray, lang=None, psm=None):
        PyTesser.check_path() #Check if tesseract available in the path
        cv2.imwrite(TEMP_IMAGE, gray)
        PyTesser.process_request(TEMP_IMAGE, TEMP_FILE, lang, psm)
        f = open(TEMP_FILE+".txt", "r") #Open back the file
        txt = f.read()
        f.close()
        os.remove(TEMP_FILE+".txt")
        os.remove(TEMP_IMAGE)
        return txt    

##
sample_db_path = "./sample/"
test_db_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Tmp/VIN/"
filename = "1.jpg"
fullpath = test_db_path + filename
###

if __name__ == "__main__":
    image = cv2.imread(fullpath)
    #showResult("image",image)
    analyzeRIL(image)
    