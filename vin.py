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
import random
import matplotlib.pyplot as plt
import time
import codecs
import types
import re

from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk

from feature.bbox import showResult
from misc.contour import String,Contour,ROI
from misc.preprocess import preprocess as ipreprocess
from misc.preprocess import maximizeContrast as icontrast

from feature.extractfeature import goodfeatures_revision
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
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5

ascii_regex = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
unicode_regex = [u'0',u'1',u'2',u'3',u'4',u'5',u'6',u'7',u'8',u'9',u'A',u'B',u'C',u'D',u'E',u'F',u'G',u'H',u'I',u'J',u'K',u'L',u'M',u'N',u'O',u'P',u'Q',u'R',u'S',u'T',u'U',u'V',u'W',u'X',u'Y',u'Z']

def Ischinese(string):
    if re.sub(r'\W', "", string) == "":
        return True
    else:
        return False

def utf2int(string):
    #utf8 = '\xe3\x80\x81' 
    utf8 = string.encode('hex')
    val = 0 
    for octet in utf8: 
        val = ( val * 256 ) + ord( octet ) 
    print val

def hex2utf8(integer):
    utf8encode = codecs.getencoder( 'utf-8' ) 
    return utf8encode(unichr(0x3001))[0] 
 
def ascii2int(ascii):
    return ord(ascii)

def int2ascii(integer):
    return unichr(integer)

def uint8tobinary(mask,reverse=False):
    mask_ = mask.copy()
    if reverse:
        mask_[mask>128] = 1
        mask_[mask<=128] = 0
    else:
        mask_[mask>128] = 0
        mask_[mask<=128] = 1
    return mask_

def maskize(img,masks,reverse=False):
    if type(masks) is types.ListType:
        mask_ = uint8tobinary(masks[0],reverse=reverse)
        for i in range(1,len(masks)):
            mask_ *= uint8tobinary(masks[i])
    else:    
        mask_ = uint8tobinary(masks,reverse=reverse)
    if len(img.shape) > 2:
        masked = img*mask_[:,:,np.newaxis]
    else:
        masked = img*mask_
    return masked

def Laplacian(gray):
    denoised = cv2.GaussianBlur(gray,(5,5),0)
    laplacian = cv2.Laplacian(denoised,cv2.CV_64F, ksize = 3,scale = 2,delta = 1)
    laplacian -= np.amin(laplacian)
    laplacian = laplacian * 255 / (np.amax(laplacian) - np.amin(laplacian))
    laplacian[laplacian>255]=255
    laplacian = laplacian.astype('uint8')
    return icontrast(laplacian)#laplacian

def Sobel(gray):
    sobel_vertical = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize = 3)
    sobel_horizontal = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    sobel = cv2.min(sobel_horizontal,sobel_vertical)
    #print np.amin(sobel),np.amax(sobel)
    min_,max_ = np.amin(sobel),np.amax(sobel)
    sobel -= min_
    sobel = sobel * 255 / (max_ - min_)
    sobel = sobel.astype('uint8')
    return sobel    

def Entropy(gray):
    en_ = entropy(gray, disk(3))
    min_ = np.amin(en_)
    max_ = np.amax(en_)
    en_ -= min_
    en_ = en_ * 255 / (max_ - min_)
    en_ = en_.astype('uint8')
    return en_

def Garbor(gray):        
    #https://corpocrat.com/2015/03/25/applying-gabor-filter-on-faces-using-opencv/
    src = gray.astype('float32')
    ksize,sigma,gamma,ps = 31, 1.0, 0.02, 0
    filters = []
    for theta in np.arange(0, np.pi, np.pi / 8):
        for lamda in np.arange(0, np.pi, np.pi/4): 
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, ps, ktype=cv2.CV_32F)
            kernel /= 1.5*kernel.sum()
            filters.append(kernel)
    dest = np.zeros((src.shape),np.float32)#np.zeros_like(src.shape)
    for kernel in filters:
        fimg = cv2.filter2D(src, cv2.CV_32F, kernel)
        np.maximum(dest, fimg, dest)
    #ksize,sigma,theta,lamda,gamma,ps = 31, 1, 0, 1.0, 0.02, 0
    #kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, ps, ktype=cv2.CV_32F)
    return dest.astype('uint8')    

def DoG(gray):    
    #equalized_image = cv2.equalizeHist(gray)
    imgb1 = cv2.GaussianBlur(gray, (11, 11), 0)
    imgb2 = cv2.GaussianBlur(gray, (31, 31), 0)
    return imgb1 - imgb2#Difference of Gaussians    

def tophatblackhat(gray):
    gray = cv2.GaussianBlur(gray,(3,3),0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    showResult("tophat",tophat)
    showResult("blackhat",blackhat)
    
def TopHat(gray):
    # initialize a rectangular (wider than it is tall) and square
    # structuring kernel
    h,w = gray.shape
    #gray=255-gray
    gray = cv2.resize(gray,None,fx=0.4,fy=0.4)
    #gray = cv2.GaussianBlur(gray,(3,3),0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # apply a tophat (whitehat) morphological operator to find light
    # regions against a dark background (i.e., the credit card numbers)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    hat = cv2.max(tophat,blackhat)
    #showResult("tophat",tophat)
    # compute the Scharr gradient of the tophat image, then scale
    # the rest back into the range [0, 255]
    gradX = cv2.Sobel(hat, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)
    #gradY = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=0, dy=1,ksize=-1)
    grad = gradX#cv2.min(gradX,gradY)
    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    grad = (255 * ((grad - minVal) / (maxVal - minVal)))
    grad = grad.astype("uint8")
    # apply a closing operation using the rectangular kernel to help
    # cloes gaps in between credit card number digits, then apply
    # Otsu's thresholding method to binarize the image
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]    
    # apply a second closing operation to the binary image, again
    # to help close gaps between credit card number regions
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, sqKernel)
    thresh = cv2.dilate(thresh,None, iterations=6)
    thresh = cv2.resize(thresh,(w,h),interpolation=cv2.INTER_LINEAR)
    return thresh

class VIN(object):
    
    def __init__(self,image=None):
        self.bgr,self.gray,self.laplacian,self.sobel,self.DoG,self.lthr,self.hist,self.contour,self.compose,self.entropy,self.garbor = None, None, None, None, None, None, None, None, None, None, None
        if image is not None:
            self.bgr = image
            self.preprocess()
        self.height,self.width = 0,0
        self.contours = []
        self.chars = []
    
    def initialize(self):
        self.bgr,self.gray,self.laplacian,self.sobel,self.DoG,self.lthr,self.hist,self.contour,self.compose,self.entropy, self.garbor = None, None, None, None, None, None, None, None, None, None, None
    
    def preprocess(self):
        gray = cv2.cvtColor(self.bgr,cv2.COLOR_BGR2GRAY)
        self.gray = icontrast(255 - gray)
        self.laplacian = Laplacian(gray)
        #self.sobel = Sobel(gray)
        #self.entropy = Entropy(gray)
        #self.garbor = Garbor(gray)
        self.DoG = DoG(gray)
        self.lthr = cv2.adaptiveThreshold(self.laplacian, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)        
        self.tophat = 255 - TopHat(gray)
        #tophatblackhat(gray)
        masks = []
        masks.append(self.tophat)
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
        showResult("laplacian",self.laplacian)     
        #showResult("thr",self.lthr)
        #showResult("contour",self.contour)
        #showResult("compose",self.compose)
        showResult("tophat",self.tophat)

    def posfiltering(self,isdebug=False):
        h, w = self.contour.shape
        chars = sorted(self.chars,key=lambda char:int(math.log(char.brH,2)))
        chars = sorted(self.chars,key=lambda char:math.log(math.pow(char.brcY,2) + math.sqrt(char.brcX),2))
        VIN.drawChars((self.height,self.width),"sorting",chars,isdebug=isdebug)
        strings = []
        string = String()
        string.add(chars[0])                
        maximum_diff_posy = chars[0].brH
        maximum_diff_pos = chars[0].brH * 10
        maximum_diff_height = chars[0].brH/3
        
        i = 0
        while i<len(chars)-2:
            
            diff_pos,diff_posy,diff_size = Contour.isconnectable(chars[i],chars[i+1])
                           
            if diff_pos < maximum_diff_pos and\
               diff_size < maximum_diff_height and\
               diff_posy < maximum_diff_posy:
                string.add(chars[i+1])
            else:
                diff_pos,diff_posy,diff_size = Contour.isconnectable(chars[i],chars[i+2])
                if diff_pos < maximum_diff_pos and\
                   diff_size < maximum_diff_height and\
                   diff_posy < maximum_diff_posy:
                    string.add(chars[i+2])
                    i += 2
                    continue
                strings.append(string)
                string = String()
                string.add(chars[i+1])
                maximum_diff_posy = chars[i+1].brH
            i += 1
                
        strings.append(string)
        strings.sort(key=lambda string:string.charcount)
        self.strings = []
        for i in range(10):
            if len(strings) - i > 1 and strings[len(strings)-i-1].getlength() > 5:
                string = strings[len(strings)-i-1]
                self.strings.append(string)
                if isdebug:
                    VIN.drawChars((self.height,self.width),"third-filtering",string.getitems(),isdebug=isdebug)
                    #roi = String.extractROI(self.bgr,string)
                    #showResult("roi",roi.imgPlate)
            else:
                break
        String.sortall(self.strings)

    def sizefiltering(self,isdebug=False):
        self.chars = VIN.find_possible_chars(self.contours,True)
        if isdebug:
            VIN.drawChars((self.height,self.width),"first-filtering",self.chars,isdebug=isdebug)

    def innerfiltering(self,isdebug=False):
        chars = []
        for char in self.chars:
            [x, y, w, h] = char.brX,char.brY,char.brW,char.brH
            isinner = False
            for char_ in self.chars:
                [xx, yy, ww, hh] = char_.brX,char_.brY,char_.brW,char_.brH
                if x > xx and y > yy and (x+w) < (xx+ww) and (y+h) < (yy+hh):
                    isinner = True
            if isinner:
                continue
            chars.append(char)
        self.chars =  chars
        if isdebug:
            VIN.drawChars((self.height,self.width),"inner-filtering",self.chars,isdebug=isdebug)
            
    def finalize(self,isdebug=False):
        string_ = []
        strings_ = []
        chars = Contour.contours2chars(self.contours)
        chars.sort(key=lambda char:char.brX)
        #vis = np.zeros((self.height,self.width),np.uint8)
        #cv2.drawContours(vis,self.contours,-1,(255,255,255),-1)
        for string in self.strings:
            height = string.charheight
            delta = height * 0.15
            eta =  height * 0.3
            string_ = String()
            #cv2.line(vis,(0,string.getcenterliney(0)),(self.width-1,string.getcenterliney(self.width-1)),(255,255,255),2)
            #showResult("vis",vis)
            for char in chars:
                if abs(char.brH - height) < eta and\
                   abs(string.getcenterliney(char.brcX) - char.brcY) < delta:
                       string_.add(char)
            if string_.charcount != 0:
                string_.sort()
                strings_.append(string_)
                chars = list(set(chars) - set(string_.chars))

        String.filtering(strings_)
        if len(strings_) == 0:
            return
        #VIN.drawChars((self.height,self.width),"final",strings_[0].getitems(),isdebug=isdebug)
        string = strings_[0]
        if string.charcount > 1:
            String.mark(self.bgr,string)
        return
        if True:
            for string in strings_:
                if string.confidence > 0.7:
                    VIN.drawChars((self.height,self.width),"final",string.getitems(),isdebug=isdebug)
                    print string.result
            
    def process(self,img=None):
        start = time.time()
        if img is not None:
            self.bgr = img
            self.height,self.width,c = img.shape
            self.preprocess()
            #self.showAll()
        # size filtering
        self.sizefiltering()
        # size filtering
        self.innerfiltering()
        # ocr filtering
        #self.ocrfiltering(True)
        # distance filtering
        self.posfiltering()
        self.confidence = self.finalize(True)
        print time.time() - start
        return self.confidence
        
    def setcontours(self,isdebug=False):
        self.img_contour, self.contours, npaHierarchy = cv2.findContours(self.compose, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # find all contours        
        if isdebug:
            cv2.drawContours(self.img_contour, self.contours, -1, SCALAR_WHITE, -1)
            print "\nstep 2 - contours = " + str(len(self.contours))
            showResult("contours",self.img_contour)
        
    @staticmethod        
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

    @staticmethod    
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


##### Test Variable
dataset_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Tmp/VIN/"
filename = "38.jpg"
fullpath = dataset_path + filename

lp = VIN()

if __name__ == "__main__":
    #main(cv2.imread(fullpath,0))
    lp.initialize()
    lp.process(img=cv2.imread(fullpath))
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