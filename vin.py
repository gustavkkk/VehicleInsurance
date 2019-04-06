#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 10:19:31 2017

@author: ubuntu

regular expression:http://www.tutorialspoint.com/python/python_reg_expressions.htm
"""
import cv2
import numpy as np
import math
#import random
#import matplotlib.pyplot as plt
#import time

from feature.bbox import showResult
from misc.contour import String,Contour#,ROI
#from misc.preprocess import preprocess as ipreprocess
from misc.preprocess import maximizeContrast as icontrast

from feature.space import Laplacian,DoG,TopHat,maskize,AdaptiveThreshold,tophatblackhat

#from feature.extractfeature import goodfeatures_revision
# threshold variables ##########################################################################
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
# colr variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)
# padding variables ##########################################################################
EXPECTED_HEIGHT = 500.0
EXPECTED_WIDTH = 768.0#768.0#1024.0

ascii_regex = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
unicode_regex = [u'0',u'1',u'2',u'3',u'4',u'5',u'6',u'7',u'8',u'9',u'A',u'B',u'C',u'D',u'E',u'F',u'G',u'H',u'I',u'J',u'K',u'L',u'M',u'N',u'O',u'P',u'Q',u'R',u'S',u'T',u'U',u'V',u'W',u'X',u'Y',u'Z']

def drawChars(shape,
              title,
              chars,
              colr=SCALAR_WHITE,
              isdebug=False):
    h,w =  shape
    ratio = 1
    vis = np.zeros((h,w*ratio,3), np.uint8)
    img_contours = np.zeros((shape), np.uint8)

    for i,char in enumerate(chars):
        cv2.drawContours(img_contours, [char.contour], -1, colr,-1)
        if isdebug:
            [x, y] = char.brX,char.brY
            cv2.putText(vis,str(i),(x*ratio,y),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,128,128),1,cv2.LINE_AA)
            contour = char.contour.copy()
            contour[:,:,0] *= ratio
            cv2.drawContours(vis, [contour], -1, colr,-1)
    if isdebug:
        showResult(title,vis)
    return img_contours

def find_possible_chars(contours,
                        isdebug=False):
    #
    filtered = []                # this will be the return value
    count = 0
    for i in range(0, len(contours)):                       # for each contour                  
        contour = Contour(contours[i])   
        if contour.checkIfPossibleChar():                   # if contour is a possible char, note this does not compare to other chars (yet) . . .
            count += 1                                      # increment count of possible chars
            filtered.append(contour)                        # and add to list of possible chars   
        
    return filtered

def innerfiltering(chars,isdebug=False):
    chars_ = []
    for i,char in enumerate(chars):
        [x, y, w, h] = char.brX,char.brY,char.brW,char.brH
        isinner = False
        for j,char_ in enumerate(chars):
            [xx, yy, ww, hh] = char_.brX,char_.brY,char_.brW,char_.brH
            if x > xx and y > yy and (x+w) < (xx+ww) and (y+h) < (yy+hh):
                isinner = True
        if isinner:
            continue
        chars_.append(char)
              
    return chars_

def string2txt(image,string):
    roi = String.extractROI(image, string, False)
    bgr = roi.cropped
    gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    laplacian = Laplacian(gray)
    lthr = cv2.adaptiveThreshold(laplacian, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    dog = DoG(gray)
    compose = maskize(lthr,dog)
    ###
    #img_contour, contours, npaHierarchy = cv2.findContours(compose, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(img_contour, contours, -1, SCALAR_WHITE, -1)
    img_contour = compose
    contours, npaHierarchy = cv2.findContours(compose, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)    
    cv2.drawContours(compose, contours, -1, SCALAR_WHITE, -1)

    chars = find_possible_chars(contours,False)
    chars = innerfiltering(chars)
    chars.sort(key=lambda char:char.brX)
    img_char = drawChars(gray.shape,"size-filtering",chars)
    compose = maskize(img_contour,255-img_char)
    #showResult("compose",compose)
    ### OCR
    ocrresult,count = string.ocr2(compose,chars)
    string.meaningfulcharcount = count
    return ocrresult
    
class VIN(object):
    
    def __init__(self,image=None):
        self.initialize()
        if image is not None:
            self.bgr = image
            self.preprocess()
    
    def initialize(self):
        self.bgr,self.gray,self.laplacian,self.sobel,self.DoG,self.lthr,self.hist,self.contour,self.compose,self.entropy, self.garbor = None, None, None, None, None, None, None, None, None, None, None
        self.height,self.width = 0,0
        self.contours = []
        self.chars = []
        self.strings = []
    
    def preprocess(self):
        gray = cv2.cvtColor(self.bgr,cv2.COLOR_BGR2GRAY)
        self.gray = icontrast(255 - gray)
        self.laplacian = Laplacian(gray)
        #self.sobel = Sobel(gray)
        #self.entropy = Entropy(gray)
        #self.garbor = Garbor(gray)
        self.DoG = DoG(gray)
        self.lthr = AdaptiveThreshold(self.laplacian)
        #self.tophat = 255 - TopHat(gray)
        #tophatblackhat(gray)
        masks = []
        #masks.append(self.tophat)
        masks.append(self.DoG)
        self.compose = maskize(self.lthr,masks)
        self.contour = np.zeros((self.compose.shape[:2]),np.uint8)
        self.setcontours()

    def showAll(self):
        showResult("img",self.bgr)  
        #showResult("gray",self.gray)
        #showResult("sobel",self.sobel)
        #showResult("entropy",self.entropy)
        showResult("DoG",self.DoG)
        #showResult("garbor",self.garbor)
        #showResult("laplacian",self.laplacian)     
        #showResult("thr",self.lthr)
        #showResult("contour",self.contour)
        showResult("compose",self.compose)
        #showResult("tophat",self.tophat)

    def posfiltering(self,isdebug=False):
        h, w = self.contour.shape
        #chars = self.chars
        chars = self.sortchars()
        drawChars((self.height,self.width),"sorting",chars,isdebug=isdebug)
        strings = []
        string = String()
        string.push(chars[0])                
        maximum_diff_posy = chars[0].brH
        maximum_diff_pos = chars[0].brH * 10
        maximum_diff_height = chars[0].brH/3
        
        i = 0
        while i<len(chars)-1:
            
            diff_pos,diff_posy,diff_size = Contour.isconnectable(chars[i],chars[i+1])
                           
            if diff_pos < maximum_diff_pos and\
               diff_size < maximum_diff_height and\
               diff_posy < maximum_diff_posy:
                string.push(chars[i+1])
            else:
                if i < len(chars) - 2:
                    diff_pos,diff_posy,diff_size = Contour.isconnectable(chars[i],chars[i+2])
                    if diff_pos < maximum_diff_pos and\
                       diff_size < maximum_diff_height and\
                       diff_posy < maximum_diff_posy:
                        string.push(chars[i+2])
                        i += 2
                        continue
                strings.append(string)
                string = String()
                string.push(chars[i+1])
                maximum_diff_posy = chars[i+1].brH
            i += 1
                
        strings.append(string)
        strings.sort(key=lambda string:string.charcount)
        self.strings = []
        for i in range(10):
            if len(strings) - i >= 1 and strings[len(strings)-i-1].getlength() > 5:
                string = strings[len(strings)-i-1]
                self.strings.append(string)
                if isdebug:
                    drawChars((self.height,self.width),"third-filtering",string.getitems(),isdebug=isdebug)
                    #roi = String.extractROI(self.bgr,string)
                    #showResult("roi",roi.imgPlate)
            else:
                break
        String.sortall(self.strings)

    def sizefiltering(self,isdebug=False):
        #
        self.chars = find_possible_chars(self.contours,True)
        #
        self.shistogramfiltering()
        if isdebug:
            drawChars((self.height,self.width),"first-filtering",self.chars,isdebug=isdebug)
  
    def shistogramfiltering(self):
        histogram = {}
        for char in self.chars:
            char.logsize = round(math.log(char.brH,2))
            if char.logsize in histogram:
                histogram[char.logsize] += 1
            else:
                histogram[char.logsize] = 1
        #histogram =  sorted(histogram,key = histogram.get,reverse=True)
        filtered = [key for key in histogram if histogram[key] < 300 and histogram[key] > 10]
        chars_ = []
        for char in self.chars:
            if char.logsize in filtered:
                chars_.append(char)
        self.chars = chars_

    def sortchars(self):
        chars = sorted(self.chars,key=lambda char:round(math.log(char.brH,2)))
        chars = sorted(self.chars,key=lambda char:math.log(math.pow(char.brcY,2) + math.sqrt(char.brcX),2))        
        self.chars = chars
        return chars
    
    def noisefiltering(self,isdebug=False):
        chars_ = []
        chars = self.chars#self.sortchars()
        for i in range(2,len(chars)-2):
            [xc, yc, h] = chars[i].brcX,chars[i].brcY,chars[i].brH
            count = 0
            for j in [-2,-1,1,2]:
                [xxc, yyc] = chars[i+j].brcX,chars[i+j].brcY
                if np.linalg.norm(np.array([xc,yc]) - np.array([xxc,yyc])) < 2*h:
                    count += 1
            if count < 1:
                continue
            chars_.append(chars[i])
        self.chars =  chars_
        if isdebug:
            drawChars((self.height,self.width),"inner-filtering",self.chars,isdebug=isdebug)
              
    def makeup(self,isdebug=False):
        string_ = []
        strings_ = []
        chars = self.chars#Contour.contours2chars(self.contours)
        chars.sort(key=lambda char:char.brX)
        #vis = np.zeros((self.height,self.width),np.uint8)
        #cv2.drawContours(vis,self.contours,-1,(255,255,255),-1)
        for string in self.strings:
            # need checking or not
            if string.confidence > 0.9 and string.density > 0.9:
                strings_.append(string)
                continue
            height = string.charheight
            delta = height * 0.15
            eta =  height * 0.3
            string_ = String()
            #cv2.line(vis,(0,string.getcenterliney(0)),(self.width-1,string.getcenterliney(self.width-1)),(255,255,255),2)
            #showResult("vis",vis)
            for char in chars:
                if abs(char.brH - height) < eta and\
                   abs(string.getcenterliney(char.brcX) - char.brcY) < delta:
                       string_.push(char)
            if string_.charcount != 0:
                string_.sort()
                strings_.append(string_)
                chars = list(set(chars) - set(string_.chars))

        strings_.sort(key=lambda string:string.confidence,reverse=True)
        if isdebug:
            for string in strings_:
                drawChars((self.height,self.width),"final",string.getitems(),isdebug=isdebug)        
        return strings_

    @staticmethod
    def ocrchecking(image,strings,isMandatory=False):
        if len(strings) == 0:
            return None
        string = strings[0]
        #string2txt(image,string)
        strResult = string2txt(image,string)
        print strResult.encode('utf-8')
        #print string.confidence,string.density,string.meaningfulcharcount
        if string.isAcceptable() is False or isMandatory:
            strings = String.filtering_by_ocr(strings)
        strings.sort(key=lambda string:string.confidence + 0.06 * string.meaningfulcharcount + string.density,reverse=True)
        strings = String.filtering_by_criteria(strings)
        if len(strings) != 0:
            return strings
        else:
            return None
        
    def finalize(self,isdebug=False):
        strings_ = self.makeup(isdebug=isdebug)
        # need ocr or not
        strings_ = VIN.ocrchecking(self.bgr,strings_,False)
        #
        if strings_ is not None:
            marked = String.mark(self.bgr,strings_)
            if isdebug:           
                showResult("marked",marked)
                    #drawChars((self.height,self.width),"final",string.getitems(),isdebug=isdebug)
                    #print string.result
            return True,strings_[0].confidence,marked
        else:
            return False,0.0,None
        
    def resize(self,img):
        h,w,c = img.shape
        if h > w:
            ratio = EXPECTED_WIDTH/w
            start = (h-w)/2
            cropped = img[start:start+w,:,:]
        else:
            ratio = EXPECTED_WIDTH/w
            cropped = img
        #ratio = 1.0 if ratio < 1.0 else ratio
        self.bgr = cv2.resize(cropped,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_CUBIC)
        self.height,self.width,c = self.bgr.shape
        
    def process(self,img=None,isdebug=False):
        #start = time.time()
        if img is not None:
            self.initialize()
            self.resize(img)
            self.preprocess()
            #self.showAll()
        self.sizefiltering()#sizefiltering
        self.chars = innerfiltering(self.chars)#innerfiltering
        if len(self.chars) == 0:
            return False,0.0,None
        #self.noisefiltering()#noisefiltering
        self.posfiltering(isdebug)#distance filtering
        #self.finalize(isdebug=isdebug)#makingup and finalizeing
        #print time.time() - start
        return self.finalize(isdebug=isdebug)
        
    def setcontours(self,isdebug=False):
        #self.img_contour, self.contours, npaHierarchy = cv2.findContours(self.compose, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # find all contours        
        self.contours, npaHierarchy = cv2.findContours(self.compose, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # find all contours        
        self.img_contour = self.compose
        if isdebug:
            cv2.drawContours(self.img_contour, self.contours, -1, SCALAR_WHITE, -1)
            print "\nstep 2 - contours = " + str(len(self.contours))
            showResult("contours",self.img_contour)

    '''
    def TestOCR(self,isdebug=False):
        strings_ = self.makeup(isdebug=isdebug)
        for string in strings_:
            ocrresult,count = string2txt(self.bgr,string)
            
    def TestSegmentation(self,isdebug=False):
        strings_ = self.makeup(isdebug=isdebug)
        String.mark(self.bgr,strings_)
        for string in strings_:
            String.makesegments(self.bgr,string)
    '''
#http://jmgomez.me/a-fruit-image-classifier-with-python-and-simplecv/        
##### Test Variable
dataset_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Tmp/VIN/"
test_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Tmp/VIN-scrapy/renamed/"
filename = "2.jpg"
fullpath = test_path + filename

vin = VIN()

import time

if __name__ == "__main__":
    #main(cv2.imread(fullpath,0))
    start = time.time()
    isFound,confidence,marked = vin.process(img=cv2.imread(fullpath),isdebug=True)
    print("total elapsed time: "+str(int((time.time() - start)*1000)/1000.0)+"s")
    #VIN.detect_by_gf(cv2.imread(fullpath))
    #VIN.detect_by_erfilter(img=cv2.imread(fullpath),isdebug=True)
    #VIN.detect_by_contour(img=cv2.imread(fullpath),isdebug=True)
    #VIN.detect_by_smer(img=cv2.imread(fullpath),isdebug=True)
  
'''
    def ocrfiltering(self,isdebug=False):
        chars = []
        h,w = self.bgr.shape[:2]
        for candidate in self.chars:
            candidate.setBinary()
            binary = candidate.getBinary()
            ocrresult = self.ocr.img2string_single(binary)
            
            if ocrresult != u"":
                integer = ord(ocrresult[0])
                if (48 <= integer and integer <= 57) or\
                   (65 <= integer and integer <= 90) or\
                   (97 <= integer and integer <= 122):
                    #print ocrresult
                    candidate.char = ocrresult
                    chars.append(candidate)
        self.chars =  chars
        if isdebug:
            VIN.drawChars((self.height,self.width),"second-filtering",self.chars,isdebug=isdebug)
            
    @staticmethod
    def detect_by_contour(img,
                          isdebug=False):
        # Initialize
        height, width, numChannels = img.shape
        img_gray = np.zeros((height, width, 1), np.uint8)
        img_thr = np.zeros((height, width, 1), np.uint8)
        img_contour = np.zeros((height, width, 3), np.uint8)
        # Grayscale
        img_gray, img_thr = ipreprocess(img)
        if isdebug:
            showResult("img_gray",img_gray)
            showResult("img_thr",img_thr)
            #showResult("Test",cv2.Canny(img_gray,50,200))
        # First Filtering(Contours2Chars)
        chars = VIN.find_possible_chars(img_thr)
        if isdebug:
            print "step 2 - the numbder of suspicious chars(roughly filtered contours) = " + str(len(chars))
            VIN.drawChars(img_contour,"first-filtering",chars,isdebug=True)
        # Second Filtering(Chars2Strings)
        strings = Contour.findListOfListsOfMatchingChars(chars)
        if isdebug: # show steps #######################################################
            print "step 3 - strings.Count = " + str(len(strings))    # 13 with MCLRNF1 image
            img_contour = np.zeros((height, width, 3), np.uint8)
    
            for string in strings:
                (b,g,r) = (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
                img_contour = VIN.drawChars(img_contour,"second-filtering",string,(b,g,r),isdebug=True)
        # Third Filtering(String2ROIs)
        ROIs = []
        bboxes = []
        for string in strings:                   # for each group of matching chars
            roi = String.extractROI(img, string)         # attempt to extract plate
    
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
    
                cv2.line(img, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
                cv2.line(img, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
                cv2.line(img, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
                cv2.line(img, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
                showResult("roi",ROIs[i].imgPlate)
    
            showResult("result",img)
    
        return bboxes

    def erfiltering(self):
        vis = np.zeros((self.bgr.shape[:2]),np.uint8)
        images = []
        channels = []
        images.append(self.gray)
        images.append(255-self.gray)
        for image in images:
            laplacian = Laplacian(image)
            thr = cv2.adaptiveThreshold(laplacian, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
            channels.append(laplacian)
        for channel in channels:
            erc1 = cv2.text.loadClassifierNM1('./model/trained_classifierNM1.xml')
            er1 = cv2.text.createERFilterNM1(erc1,16,0.00015,0.13,0.2,True,0.1)            
            erc2 = cv2.text.loadClassifierNM2('./model/trained_classifierNM2.xml')
            er2 = cv2.text.createERFilterNM2(erc2,0.5)            
            regions = cv2.text.detectRegions(channel,er1,er2)                
            cv2.drawContours(vis, regions, -1, (255,255,255),-1)
        showResult("vis",vis)
            
    @staticmethod
    def detect_by_smer(img,
                       isdebug=False):
        gray, thr = ipreprocess(img)
        gray = 255 - gray
        showResult("gray",gray)
        # found suspicious regions
        mser = cv2.MSER_create(2,60,1200)
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

            rects = regions[1]
            #print rects
            for rect in rects:
                x,y,w,h = rect
                print rect
                print x,y,w,h
                #cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),-1)
                cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),1,8,0)

            showResult("vis",vis)
        # create text region mask
        mask = np.zeros((h,w,3), dtype=np.uint8)
        mask[mask>=0]=255
        for contour in hulls_:
            cv2.drawContours(mask, [contour], -1, (0, 0, 0), -1)
        
        showResult("textmask",mask)
        return mask[:,:,1]

    @staticmethod
    def detect_by_erfilter(img,isdebug=False):       
        # for visualization
        vis = img.copy()       
        
        # Extract channels to be processed individually
        channels = cv2.text.computeNMChannels(img)
        # Append negative channels to detect ER- (bright regions over dark background)
        cn = len(channels)-1
        for c in range(0,cn):
            showResult("channel"+str(c),255-channels[c])
            channels.append((255-channels[c]))
        
        # Apply the default cascade classifier to each independent channel (could be done in parallel)
        print("Extracting Class Specific Extremal Regions from "+str(len(channels))+" channels ...")
        print("    (...) this may take a while (...)")
        for channel in channels:
        
            erc1 = cv2.text.loadClassifierNM1('./model/trained_classifierNM1.xml')
            er1 = cv2.text.createERFilterNM1(erc1,16,0.00015,0.13,0.2,True,0.1)
            
            erc2 = cv2.text.loadClassifierNM2('./model/trained_classifierNM2.xml')
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
 
    @staticmethod
    def detect_by_gf(origin,
                     isdebug=False):
        if origin is None:
            return None
        # Default Size
        h,w,c = origin.shape
        # Extract Good Features
        goodfeatures_revision(origin,True)
'''