#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 10:54:30 2017

@author: ubuntu
"""

import cv2
import numpy as np
import math
import statistics
import scipy

from feature.bbox import showResult
from feature.colorspace import checkBlue,checkYellow,rgb2hsv
from feature.space import Laplacian,DoG,AdaptiveThreshold,maskize#,TopHat
from misc.preprocess import maximizeContrast as icontrast

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 60

EXPECTED_CHAR_NUMBER = 7

PLATE_SIZE_BIG_WIDTH = 160
PLATE_SIZE_BIG_HEIGHT = 44
PLATE_SIZE_SMALL_WIDTH = 136
PLATE_SIZE_SMALL_HEIGHT = 36
PLATE_ROTATE_MARGIN = 30

SMALL_GAP_MAXIMUM = 20
BIG_GAP_MAXIMUM = 28
SMALL_GAP_MINIMUM = 10
BIG_GAP_MINIMUM = SMALL_GAP_MINIMUM + 2
BIG_INTERVAL = 6
SMALL_INTERVAL = 6
# trend detection
def linreg(X, Y):
    """
    return a,b in solution to y = ax + b such that root mean square distance between trend line and original points is minimized
    """
    N = len(X)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in zip(X, Y):
        Sx = Sx + x
        Sy = Sy + y
        Sxx = Sxx + x*x
        Syy = Syy + y*y
        Sxy = Sxy + x*y
    det = Sxx * N - Sx * Sx
    return (Sxy * N - Sy * Sx)/det, (Sxx * Sy - Sx * Sxy)/det

colrs = {0:"Blue",1:"Yellow"}

class LicensePlate(object):
    
    def __init__(self,image=None):
        self.initialize()
        if image is not None:
            self.originheight,self.originwidth = image.shape[:2]
            self.bgr = cv2.resize(image, (self.width, self.height),interpolation = cv2.INTER_CUBIC)
            self.preprocess()

    def initialize(self):
        self.bgr,self.gray,self.denoised,self.laplacian,self.thr,self.DoG,self.returnImg,self.compose = None, None, None, None, None, None, None, None
        self.height,self.width = PLATE_SIZE_BIG_HEIGHT,PLATE_SIZE_BIG_WIDTH
        self.contours = []
        self.charwidth = 0.0
        self.charheight = 0.0
        self.charangle = 0.0
        self.chargaps = []
        self.charwidths = []
        self.charheights = []
        self.charcenters = []
        self.charbiggap = 0
        self.charsmallgap = 0
        self.charleftmost = 0
        self.charrightmost = 0
        self.charsegpoints = []
        #
        self.colr = 0#blue:1 or yellow:2
        self.segImgs = []
        self.confidence = 0.0
        #
        self.mode = "Blue"
        #
        self.x1y1,self.x0y1,self.x1y0, self.x0y0 =0,0,0,0
        
    def preprocess(self,mode="Blue"):
        gray = cv2.cvtColor(self.bgr,cv2.COLOR_BGR2GRAY)
        if mode == "Blue":
            self.mode = "Blue"
            self.gray = gray#iicontrast(gray)#contrast(gray)#equalized_image = cv2.equalizeHist(self.gray)
        else:
            self.mode = "Yellow"
            self.gray = 255 - gray
        self.DoG = DoG(self.gray)#Difference of Gaussians
        self.laplacian = Laplacian(self.gray,needcontrast=False)
        self.thr = AdaptiveThreshold(self.laplacian)
        self.thr = maskize(self.thr,self.DoG)
        #self.tophat = 255 - TopHat(gray)
        #self.DoG = cv2.dilate(self.DoG,(3,3),iterations=3)

    def makeCompose(self):
        self.gray = cv2.cvtColor(self.resultimg,cv2.COLOR_BGR2GRAY)
        if self.mode == "Yellow":
            self.gray = 255 - self.gray
        # DOG
        equalized_image = cv2.equalizeHist(self.gray)
        self.DoG = DoG(equalized_image)#Difference of Gaussians
        self.compose = maskize(self.thr,self.DoG)
               
    def showAll(self):
        showResult("roi",self.bgr)
        showResult("gray",self.gray)        
        showResult("laplacian",self.laplacian)
        showResult("DoG",self.DoG)
        showResult("thr",self.thr)
        #showResult("tophat",self.tophat)

    def debug(self,img,xs):
        tmp = img.copy()
        for x in xs:
            cv2.circle(tmp,(int(x),self.height/2),2,(255,0,0),2)
        showResult("debug",tmp)
        
    def drawChars(self,
                  title,
                  colr=SCALAR_WHITE,
                  isdebug=False):
        img_contours = np.zeros((self.bgr.shape), np.uint8)
        cv2.drawContours(img_contours, self.contours, -1, colr,-1)
        if isdebug:
            showResult(title,img_contours)
        return img_contours
        
    def sort_contours_by_poistion(self):
        '''
        matches = []
        for contour in self.contours:
            [x, y, w, h] = cv2.boundingRect(contour)
            matches.append([x,contour])
            
        matches = sorted(matches,reverse=False)
        
        contours = []
        for match in matches:
            contours.append(match[1])
            
        self.contours = contours
        '''
        self.contours.sort(key=lambda contour:cv2.boundingRect(contour)[0])
        
    def detectcolr(self):
        c_x = self.width / 2
        c_y = self.height / 2
        roi = self.bgr[c_y - 15:c_y + 15,c_x-30 :c_x+30,:]
        rgb = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
        h,w,c = roi.shape
        totalcolr = np.zeros(3)
        for j in range(h):
            for i in range(w):
                totalcolr += rgb[j,i]
        avergecolr = totalcolr / (w * h)
        hsv = rgb2hsv(avergecolr)
        if checkBlue(hsv):
            self.colr = 1
            self.returnImg =  255 - self.laplacian
        elif checkYellow(hsv):
            self.colr = 2
            self.returnImg = self.laplacian.copy()
        else:
            self.colr = 0

    def filtering_contours(self,isdebug=False):
        #first filtering
        if self.first_filtering(isdebug=isdebug) is None:
            return False
        if isdebug:
            self.drawChars("first-filtering",isdebug=isdebug)
        #second filtering
        if self.second_filtering() is None:
            return False
        if isdebug:
            self.drawChars("second-filtering",isdebug=isdebug)
        #third filtering including sorting contours by x position
        if self.third_filtering() is None:
            return False
        if isdebug:
            self.drawChars("third-filtering",isdebug=isdebug)
        return True
        
    def process(self,
                roi=None,
                mode="Blue",
                isdebug=False):
        self.initialize()
        if roi is not None:
            self.originheight,self.originwidth = roi.shape[:2]
            self.bgr = cv2.resize(roi, (self.width, self.height),interpolation = cv2.INTER_CUBIC)
            self.preprocess(mode=mode)
        #
        if isdebug:
            self.showAll()
            
        # filtering contours
        if self.filtering_contours(isdebug=isdebug) is False:
            return 0.0
        #
        #https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/
        #https://docs.opencv.org/2.4/modules/core/doc/basic_structures.html?highlight=rotatedrect#RotatedRect::RotatedRect()
        #
        # calculate x range of LP        
        self.estimateLP(isdebug=isdebug)
        if isdebug:
            tmp = self.bgr.copy()
            cv2.line(tmp,(self.charleftmost,self.height/2),(self.charrightmost,self.height/2),(34,222,0),2)
            showResult("test",tmp)
        # crop and warp, then segment
        self.correctImg(isdebug=isdebug)    
        # detect colr
        #self.detectcolr()
        # makecompose
        self.makeCompose()
        # makesegments
        self.makesegments(self.compose)
        # calculate confidence
        self.calculateconfidence()
        return self.confidence

    #
    # 1.filter by gaps between contours and height info between neighbored contours
    # 2.calculate smallgap,biggap of contours to be used for estimating LP
    #
    def third_filtering(self):
        #
        cur_contour_count = len(self.contours)
        if cur_contour_count == 0:
            return [],[],[],[]
        elif cur_contour_count == 1:
            [x, y, w, h] = cv2.boundingRect(self.contours[0])
            return [w+1],[w],[h],[x+w/2]
        # sort by increasing order
        self.sort_contours_by_poistion()
        #
        gaps = []
        contours = []
        widths = []
        heights = []
        centers = []
        for i in range(cur_contour_count-1):
            [x, y, w, h] = cv2.boundingRect(self.contours[i])
            x_c = x + w/2
            y_c = y + h/2
            [xx, yy, ww, hh] = cv2.boundingRect(self.contours[i+1])
            xx_c = xx + ww/2
            yy_c = yy + hh/2
            gap = xx_c - x_c
            #print gap
            if gap > SMALL_GAP_MINIMUM and gap < BIG_GAP_MAXIMUM:
                gaps.append(gap)
            elif gap >= BIG_GAP_MAXIMUM:
                #interval = float(gap - w/2 - ww/2)
                #count = round(gap/self.charwidth) + 1
                count = round(gap/float(self.charwidth))
                gap /= count
                gaps.append(gap)
            else:
                continue
            contours.append(self.contours[i])
            widths.append(w)
            heights.append(h)
            centers.append(x_c)
            if i == (cur_contour_count-2) and (abs(yy_c - y_c) + abs(yy-y)) < h*0.15:
                contours.append(self.contours[i+1])
                widths.append(ww)
                heights.append(hh)
                centers.append(xx_c)
                
        if len(widths) == len(gaps):
            gaps.pop()

        if len(contours) < 2:
            return None
            
        self.contours = contours
        self.chargaps = gaps
        self.charwidths = widths
        self.charheights = heights
        self.charcenters = centers
        
        return self.chargaps,self.charwidths,self.charheights,self.charcenters
                
    def setcharcenters(self,centers):
        delta = 3
        for time in range(3):
            if len(centers) == EXPECTED_CHAR_NUMBER:
                break
            for i in range(1,len(centers)):
                dis = centers[i] - centers[i-1]
                if dis > (self.charsmallgap*2 - delta):
                    if i == 2:
                        centers.insert(i,centers[i-1]+self.charbiggap)
                    else:
                        centers.insert(i,centers[i-1]+self.charsmallgap)
                    break
        self.charcenters = sorted(centers)

    def setcharsegpoints(self):
        self.charsegpoints.append(self.charleftmost)
        for i in range(1,EXPECTED_CHAR_NUMBER):
            self.charsegpoints.append((self.charcenters[i-1] + self.charcenters[i])/2)
        self.charsegpoints.append(self.charrightmost)
        
    def estimateLP(self,isdebug=False):
        #
        cur_contour_count = len(self.contours)
        if cur_contour_count < 2:
            return 0,0
        # calculate gaps,widths,heights of final contours
        gaps,widths,heights,centers = self.chargaps,self.charwidths,self.charheights,self.charcenters
        biggap = gaps[0]
        smallgap,smallergap = statistics.median_high(gaps),statistics.median_low(gaps)
        if smallgap == biggap or smallgap == max(gaps):
            smallgap = smallergap
        if biggap >= (smallgap+BIG_INTERVAL) and biggap < 2.3 * smallgap:
            isbgFound =  True
        else:
            isbgFound = False
            biggap = BIG_INTERVAL + smallgap#+= BIG_INTERVAL
        self.charbiggap = biggap
        self.charsmallgap = smallgap
        # detect trend
        a,b = linreg(range(len(heights)),heights)
        #            
        width_leftmost = widths[0]
        width_rightmost = widths[cur_contour_count-1]
        rightmost = max(centers)
        leftmost = min(centers)
        # calculate centers, x position of license plate
        char_count_present =  int(float(rightmost-leftmost-biggap) / smallgap+0.2) + 2 if isbgFound else\
                              int(float(rightmost-leftmost) / smallgap+0.2) + 1
        char_count_present = int(max(char_count_present,cur_contour_count))
        #print "estimateLP:char_count_present",char_count_present
        #char_count_present = 6 if char_count_present > 6 else char_count_present
        # fill missing centers where no biggap,but pretty bigger than smallgap
        while char_count_present > cur_contour_count and isbgFound is False:
            for i in range(len(gaps)):
                if gaps[i] > 1.7*smallgap:# and abs(centers[i]+smallgap - centers[i+1]) < smallgap/2:
                    cur_contour_count += 1
                    tmp = gaps[i]
                    gaps[i] = smallgap
                    gaps.insert(i+1,tmp-smallgap)
                    centers.insert(i+1,centers[i]+smallgap)
                    break
            break
        while(char_count_present < EXPECTED_CHAR_NUMBER):
            if char_count_present == 6:
                #if isbgFound is False:
                #    centers.insert(1,leftmost+biggap)
                if a > 0:
                    leftmost -= smallergap
                else:
                    leftmost -= max([width_leftmost,smallgap,self.charwidth])
                centers.insert(0,leftmost)
                char_count_present += 1
                break
            else:
                if isbgFound:
                    leftmost -= max([width_leftmost,self.charwidth])
                    centers.insert(0,leftmost)
                    for i in range(EXPECTED_CHAR_NUMBER - char_count_present - 1):
                        rightmost += smallgap
                        centers.append(rightmost)
                        char_count_present += 1
                else:
                    count = 0
                    if char_count_present != 5:
                        while((rightmost+smallgap+width_rightmost/2)<self.width):
                            count += 1
                            rightmost += smallgap
                            centers.insert(0,rightmost)
                            char_count_present += 1
                        for i in range(EXPECTED_CHAR_NUMBER - count - char_count_present - 2):     
                            leftmost -= smallgap
                            centers.insert(0,leftmost)
                            char_count_present += 1
                    leftmost -= biggap
                    centers.insert(0,leftmost)
                    char_count_present += 1
                    leftmost -= smallgap
                    centers.insert(0,leftmost)
                    char_count_present += 1
                break 
        while(len(centers) < EXPECTED_CHAR_NUMBER):
            print len(centers)
            if char_count_present == EXPECTED_CHAR_NUMBER and len(centers) == (EXPECTED_CHAR_NUMBER-1):
                if self.chargaps[0] < smallergap*1.7:
                    if a > 0:
                        leftmost -= smallergap
                    else:
                        leftmost -= max([width_leftmost,smallgap,self.charwidth])
                    centers.insert(0,leftmost)
                else:
                    centers.insert(1,leftmost+smallergap)
            break
                
        ratio = 1.85 + 4 * abs(math.cos((self.charangle/180.0) * math.pi))
        leftmost = int(leftmost-width_leftmost/ratio) if a < 0 else int(leftmost-smallgap/ratio)
        rightmost = int(rightmost+width_rightmost/ratio) if a > 0 else int(rightmost+smallgap/ratio)             
        self.charleftmost,self.charrightmost = leftmost,rightmost
        #
        self.setcharcenters(centers)
        if isdebug: 
            self.debug(self.bgr,self.charcenters)
        self.setcharsegpoints()
        if isdebug:
            self.debug(self.bgr,self.charsegpoints)
        #
        return leftmost,rightmost
    
    #
    # 1.filter by size and ratio
    # 2.calculate medianhegiht, medianwidth of contours to be used for next filtering
    #
    #@staticmethod
    def first_filtering(self,
                        isdebug=False):
        
        #img_contour, contours, npaHierarchy = cv2.findContours(self.thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # find all contours
        contours, npaHierarchy = cv2.findContours(self.thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        height, width = self.thr.shape
        img_contour = np.zeros((height, width, 3), np.uint8)
        
        cheights = []
        cwidths = []
        for contour in contours:
    
            if isdebug:
                cv2.drawContours(img_contour, [contour], -1, SCALAR_WHITE)
                
            [x, y, w, h] = cv2.boundingRect(contour)
            braRatio = float(w) / float(h)#aspect ratio
            
            if (w*h > MIN_PIXEL_AREA and\
                w > MIN_PIXEL_WIDTH and\
                h > self.height*0.4 and\
                0.04 < braRatio and\
                braRatio < 1.25):
                self.contours.append(contour)
                cwidths.append(w)
                cheights.append(h)

        if len(cwidths) < 2:
            return None
        
        self.charheight = statistics.median_high(cheights)
        self.charwidth = statistics.median_high(cwidths)
        if isdebug:
            print "\nfiltering step 1 - contours = " + str(len(contours))
            showResult("contours",img_contour)
        return self.contours

    #
    # 1.filter by medianhegiht, medianwidth
    # 2.calculate medianangle of contours to be used for correcting image
    #
    def second_filtering(self):
        
        delta_y = 5
        delta_x = 10
        cangles = []
        contours_ = []
        contours = []
        #
        for contour in self.contours:
            [x, y, w, h] = cv2.boundingRect(contour)
            if abs(h - self.charheight) < delta_y\
              and abs(w - self.charwidth) < delta_x:
                contours_.append(contour)
                rotated_rect = cv2.minAreaRect(contour)
                #print rotated_rect
                blob_angle_deg = rotated_rect[2]
                if rotated_rect[1][0] < rotated_rect[1][1]:
                    blob_angle_deg = 90 + blob_angle_deg
                #mapMatrix = getRotationMatrix2D(center, blob_angle_deg, 1.0)
                cangles.append(blob_angle_deg)
        #
        for contour in contours_:
            [x, y, w, h] = cv2.boundingRect(contour)
            isinner = False
            for contour_ in contours_:
                [xx, yy, ww, hh] = cv2.boundingRect(contour_)
                if x > xx and y > yy and (x+w) < (xx+ww) and (y+h) < (yy+hh):
                    isinner = True
            if isinner:
                continue
            contours.append(contour)        

        if len(contours) < 2:
            return None
        #                
        self.contours = contours
        # delete all duplicate angles
        cangles_ = []
        for angle in cangles:
            if angle not in cangles_:
                cangles_.append(angle)
        self.charangle = statistics.median_high(cangles_)

        return self.contours
         
    @staticmethod
    def fitLine_ransac(pts,
                       leftx = 0,
                       rightx = PLATE_SIZE_BIG_WIDTH,
                       zero_add = 0):
        if len(pts)>=2:
            if len(pts)==2:
                avg = (pts[0][1] + pts[1][1]) / 2
                pts[0][1] = pts[1][1] = avg
                
            [vx, vy, x, y] = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01)
            lefty = int(((leftx - x) * vy / vx) + y)
            righty = int(((rightx - x) * vy / vx) + y)
            return lefty+30+zero_add,righty+30+zero_add
        return 0,0

    def refinesegpts(self):
        ratio = float(self.charrightmost - self.charleftmost) / PLATE_SIZE_BIG_WIDTH
        interval = int(0.7*self.charsmallgap/ratio)
        #delta = self.height/2/ math.tan((self.charangle/180.0) * math.pi)
        ptxs = []
        for i in range(len(self.charsegpoints)):
            ptx = self.charsegpoints[i]
            ptx -= self.charleftmost
            ptx /= ratio
            
            if i > 0 and i < 7 and i!=2:
                #ptx += delta
                isFound = False
                for j in range(interval):
                    ptx_ = int(ptx - interval/2 + j)
                    if ptx_ == PLATE_SIZE_BIG_WIDTH:
                        break
                    yline = self.compose[:,ptx_]
                    if (yline == 255).sum() == 0:
                        isFound = True
                        break
                if isFound:
                    ptx = ptx_

            ptxs.append(int(ptx))

        self.charsegpoints = ptxs
            
    def makesegments(self,img):
        #
        self.refinesegpts()
        #
        #ratio = float(self.charrightmost - self.charleftmost) / PLATE_SIZE_BIG_WIDTH
        ptxs = self.charsegpoints
        #delta = (self.charbiggap - self.charsmallgap)/ratio
        #
        for i in range(EXPECTED_CHAR_NUMBER):
            start = ptxs[i]
            end = ptxs[i+1]
            '''
            if i == 1:
                end = ptxs[i] + int(self.charsmallgap / ratio)
            elif i == 2:
                start = ptxs[i+1] - int(self.charsmallgap / ratio)
            '''

            if len(img.shape) == 2:
                roi = img[:,start:end]
            else:
                roi = img[:,start:end,:]
                
            self.segImgs.append(roi)
            #showResult("roi",roi)

    @staticmethod
    def letterornot(img):
        h,w = img.shape[:2]
        hist = {}
        count = 0.0
        MINIMUM_ACCEPTABLE = 2
        for i in range(h):
            xline = img[i,:]
            nonzeros = (xline == 255).sum()
            hist[i] = nonzeros
            if nonzeros > MINIMUM_ACCEPTABLE:
                count += 1.0
                
        confidence = (img == 255).sum() / float(h*w/10)
        confidence *= count/(h/3)
        return confidence

    def calculateconfidence(self):
        sum_=0
        for i in range(EXPECTED_CHAR_NUMBER):
            confidence = self.letterornot(self.segImgs[i])
            confidence = min(1.0,confidence)
            sum_+= confidence
        self.confidence = int(sum_/EXPECTED_CHAR_NUMBER*1000)/1000.0

    def refineROI(self,pts):
        [x1y1,x0y1,x1y0, x0y0] = pts
        xratio = float(self.originwidth)/self.width
        yratio = float(self.originheight)/self.height
        self.x0y0 = [x0y0[0]*xratio,(x0y0[1] - PLATE_ROTATE_MARGIN)*yratio]
        self.x1y0 = [x1y0[0]*xratio,(x1y0[1] - PLATE_ROTATE_MARGIN)*yratio]
        self.x0y1 = [x0y1[0]*xratio,(x0y1[1] - PLATE_ROTATE_MARGIN)*yratio]
        self.x1y1 = [x1y1[0]*xratio,(x1y1[1] - PLATE_ROTATE_MARGIN)*yratio]

    def getRefinedROI(self):
        return [self.x0y0,self.x1y0,self.x1y1,self.x0y1]
    
    def correctImg(self,
                   isdebug=False):

        line_upper  = []
        line_lower = []
    
        for contour in self.contours:
            #[x,y,w,h] = cv2.boundingRect(contour)
            #line_upper.append([x,y])
            #line_lower.append([x+w,y+h])
            #leftmost = tuple(contour[contour[:,:,0].argmin()][0])
            #rightmost = tuple(contour[contour[:,:,0].argmax()][0])
            topmost = tuple(contour[contour[:,:,1].argmin()][0])
            bottommost = tuple(contour[contour[:,:,1].argmax()][0])
            line_upper.append(topmost)
            line_lower.append(bottommost)
        #
        leftx,rightx = self.charleftmost,self.charrightmost
        #
        bgr = cv2.copyMakeBorder(self.bgr,PLATE_ROTATE_MARGIN,PLATE_ROTATE_MARGIN,0,0,cv2.BORDER_REPLICATE)
        thr = cv2.copyMakeBorder(self.thr,PLATE_ROTATE_MARGIN,PLATE_ROTATE_MARGIN,0,0,cv2.BORDER_REPLICATE)
        #contourimg
        rows,cols = bgr.shape[:2]
        leftyB, rightyB = self.fitLine_ransac(np.array(line_lower),leftx,rightx,2)
        leftyU, rightyU = self.fitLine_ransac(np.array(line_upper),leftx,rightx,-2)    
        #
        rightH = rightyB - rightyU
        rightDelta = rightH / math.tan((self.charangle/180.0) * math.pi)
        leftH =  leftyB - leftyU
        leftDelta = leftH / math.tan((self.charangle/180.0) * math.pi)
       
        if self.charangle < 0:
            x1y1 = [rightx + rightDelta/2, rightyB]
            x1y0 = [rightx - rightDelta/2, rightyU]
            x0y1 = [leftx + leftDelta/2, leftyB]
            x0y0 = [leftx - leftDelta/2, leftyU]
            pts_map1  = np.float32([x1y1,x0y1,x1y0, x0y0])
        else:
            x1y1 = [rightx + rightDelta/2, rightyB]
            x1y0 = [rightx - rightDelta/2, rightyU]
            x0y1 = [leftx + leftDelta/2, leftyB]
            x0y0 = [leftx - leftDelta/2, leftyU]
            pts_map1  = np.float32([x1y1,x0y1,x1y0, x0y0])                   
 
        self.refineROI([x1y1,x0y1,x1y0, x0y0])
        x1y1 = [PLATE_SIZE_BIG_WIDTH,PLATE_SIZE_BIG_HEIGHT]
        x1y0 = [PLATE_SIZE_BIG_WIDTH,0]
        x0y1 = [0,PLATE_SIZE_BIG_HEIGHT]
        x0y0 = [0,0]
        pts_map2 = np.float32([x1y1,x0y1,x1y0, x0y0]) 
        #
        mat = cv2.getPerspectiveTransform(pts_map1,pts_map2)
        #
        self.resultimg = cv2.warpPerspective(bgr,mat,(PLATE_SIZE_BIG_WIDTH,PLATE_SIZE_BIG_HEIGHT))
        self.thr = cv2.warpPerspective(thr,mat,(PLATE_SIZE_BIG_WIDTH,PLATE_SIZE_BIG_HEIGHT))
               
    @staticmethod
    def p2abs(point):
        return math.sqrt(point[0] ** 2 + point[1] ** 2)
    
    @staticmethod
    def rotatePoint(point, angle):
        s, c = math.sin(angle), math.cos(angle)
        return (point[0] * c - point[1] * s, point[0] * s + point[1] * c)
    
    @staticmethod
    def rotatePoints(points, angle):
        return [LicensePlate.rotatePoint(point, angle) for point in points]
    
    @staticmethod
    def contour2rect(contour):
        points = map(lambda x: tuple(x[0]), contour)
        convexHull = map(lambda x: points[x], scipy.spatial.ConvexHull(np.array(points)).vertices)
        
        minArea = float("inf")
        minRect = None
        
        for i in range(len(convexHull)):
            a, b = convexHull[i], convexHull[i - 1]
            ang = math.atan2(b[0] - a[0], b[1] - a[1])
        
            rotatedHull = LicensePlate.rotatePoints(convexHull, ang)
        
            minX = min(map(lambda p: p[0], rotatedHull))
            maxX = max(map(lambda p: p[0], rotatedHull))
            minY = min(map(lambda p: p[1], rotatedHull))
            maxY = max(map(lambda p: p[1], rotatedHull))
        
            area = (maxX - minX) * (maxY - minY)
        
            if area < minArea:
                minArea = area
        
                rotatedRect = [(minX, minY), (minX, maxY), (maxX, maxY), (maxX, minY)]
                minRect = LicensePlate.rotatePoints(rotatedRect, -ang)
        
        _, topLeft = min([(LicensePlate.p2abs(p), i) for p, i in zip(range(4), minRect)])
        rect = minRect[topLeft:] + minRect[:topLeft]
        
        return rect
    '''
    def estimateLP(self):
        centers_left = []
        centers_right = []
        widths = []
        heights = []
        cur_x = 1000
        cur_x_ = 0
        width_leftmost = 0
        width_rightmost = 0
        if len(self.contours) == 0:
            return 0,self.width
        self.sort_contours_by_poistion()
        for i in range(len(self.contours)):
            [x, y, w, h] = cv2.boundingRect(self.contours[i])
            brcX = x + w / 2
            widths.append(w)
            heights.append(h)
            if brcX > self.width/2:
                #or i > 1:
                centers_right.append(brcX)
            if brcX < self.width/2:
                #or i < 2:
                centers_left.append(brcX)
            if cur_x > x:
                width_leftmost = w
                cur_x = x
            if cur_x_ < x:
                width_rightmost = w
                cur_x_ = x
        max_width = max(widths)
        #right five letters interval
        listsize = len(centers_right)
        if listsize >= 2:#center-oriented
            i = 1
            while(i < listsize and abs(centers_right[i]-centers_right[i-1]) == 0):
                i += 1
            smallgap = abs(centers_right[i] - centers_right[i-1])
            count = math.ceil(smallgap / max_width)
            print count,smallgap,centers_right[i],centers_right[i-1]
            if count >= 1 :
                smallgap /= count
            else:
                smallgap = abs(centers_right[0] - centers_left[len(centers_left)-1])
                count = math.ceil(smallgap / max_width)
                if count >= 1 :
                    smallgap /= count
                    smallgap -= 1.0                
        elif listsize == 1 and len(centers_left) > 0:
            smallgap = abs(centers_right[0] - centers_left[len(centers_left)-1])
            count = math.ceil(smallgap / max_width)
            if count >= 1 :
                smallgap /= count
                smallgap -= 1.0
        else:
            return 0,self.width
        #left second letter and third letter
        listsize = len(centers_left)
        if listsize >= 2:#center-oriented
            biggap = abs(centers_left[0] - centers_left[1])
            if biggap > smallgap*2:
                count = math.ceil(biggap / max_width)
                biggap -= (count-1)*smallgap
 
            biggap = biggap if biggap > (smallgap+2) else smallgap+3
        else:
            biggap = smallgap + 3.0
            
        print biggap,smallgap
        # found rightest letter
        rightmost = max(centers_right) if len(centers_right) > 0 else max(centers_left)
        leftmost = min(centers_left) if len(centers_left) > 0 else min(centers_right)
        char_count_present =  math.ceil((rightmost-leftmost) / smallgap) + 1 if biggap<smallgap*1.2 else\
                              math.ceil((rightmost-leftmost-biggap) / smallgap) + 2
        print char_count_present
        if char_count_present == 6:
            leftmost -= max([width_leftmost,smallgap,self.charwidth])#(smallgap - 3.0)
            char_count_present += 1
        elif char_count_present == 5:           
            if (rightmost + smallgap + width_rightmost/2 + 1) < self.width and\
                biggap > 1.25*smallgap:
                rightmost += smallgap
                leftmost -= max([width_leftmost,smallgap,self.charwidth])
                char_count_present += 1
            else:
                leftmost -= (smallgap+biggap-3.0)
        return int(leftmost-smallgap/2),int(rightmost+width_rightmost/2)
    '''
