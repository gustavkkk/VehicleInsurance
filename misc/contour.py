# self.py

import cv2
import numpy as np
import math

from feature.bbox import contour2fitline,contour2bbox,showResult
# Char Contour Filtering Definition
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8
MAX_PIXEL_WIDTH = 40
MAX_PIXEL_HEIGHT = 60

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.2

MIN_PIXEL_AREA = 40

# constants for comparing two chars
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

# other constants
MIN_NUMBER_OF_MATCHING_CHARS = 16#3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100
###################################################################################################
class Contour(object):

    # constructor #################################################################################
    def __init__(self, _contour, char=None):
                                
        self.contour = _contour

        self.boundingRect = cv2.boundingRect(_contour)
        [x, y, w, h] = self.boundingRect

        self.brX = x
        self.brY = y
        self.brW = w
        self.brH = h
        self.brA = w * h
        self.brcX = x + w / 2
        self.brcY = y + h / 2
        self.brdS = math.sqrt((w ** 2) + (h ** 2))#diagonal size
        self.braRatio = float(w) / float(h)#aspect ratio
        
        self.contourArea = cv2.contourArea(_contour)
        self.extent = float(self.contourArea)/self.brA
        #self.hull = cv2.convexHull(_contour)
        #self.hull_area = cv2.contourArea(self.hull)
        #self.solidity = float(self.contourArea)/self.hull_area        
        self.epsilon = cv2.arcLength(_contour,True)
        self.approx = cv2.approxPolyDP(_contour,0.04*self.epsilon,True)
        #
        self.binary =  np.zeros((h/4*4+4,w/4*4+4),np.uint8)
        self.contour_ = _contour.copy()
        self.char = ''
        #
        self.bbox = []
        [self.vx, self.vy, self.x, self.y] = [0,0,0,0]
        #
        self.logsize = 0

    def setfitline(self):
        [self.vx, self.vy, self.x, self.y],self.bbox = contour2fitline(self.contour)
        

    def getfitline(self):
        #print [self.vx, self.vy, self.x, self.y],self.bbox
        return [self.vx, self.vy, self.x, self.y],self.bbox
    
    def setBinary(self):
        contour_ = []
        for pt in self.contour_:
            pt[0][0] -= (self.brX - 2)
            pt[0][1] -= (self.brY - 2)
            contour_.append(pt)
        contour_ = np.array(contour_)
        h,w = self.binary.shape
        cv2.drawContours(self.binary, [contour_], -1, (255,255,255),-1)
        if h < 24:
            ratio = 24.0/h
            self.binary = cv2.resize(self.binary,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_CUBIC)
            #contour_ = contour_.astype("float")
            #contour_ *= ratio
            #contour_ = contour_.astype("int")
            
        #showResult("letter",self.binary)
        
    def getBinary(self):
        return self.binary
        
    def setImageSize(self,image):
        self.image_height,self.image_width = image.shape[:2]
        
    def checkIfPossibleChar(self):
        # this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
        # note that we are not (yet) comparing the char to other chars to look for a group
        if (self.brA > MIN_PIXEL_AREA and
            self.brW > MIN_PIXEL_WIDTH and self.brH > MIN_PIXEL_HEIGHT and
            self.brW < MAX_PIXEL_WIDTH and self.brH < MAX_PIXEL_HEIGHT and
            MIN_ASPECT_RATIO < self.braRatio and self.braRatio < MAX_ASPECT_RATIO):
            if len(self.approx) <= 10:# and self.extent < 0.6:
                return True
            else:
                return False
            return True
        else:
            return False
    '''        
    def checkIfPossibleCharInLP(self):
        if (self.brA > MIN_PIXEL_AREA and
            self.brW > MIN_PIXEL_WIDTH and self.brH > MIN_PIXEL_HEIGHT and
            0.05 < self.braRatio and self.braRatio < 1.2):
            return True
        else:
            return False
    '''
    @staticmethod 
    def contours2chars(contours):
        chars = []
        for contour in contours:
            char = Contour(contour)
            if char.brW > MIN_PIXEL_WIDTH and char.brH > MIN_PIXEL_HEIGHT:
                chars.append(char)
        return chars
    
    @staticmethod        
    def isconnectable(firstChar, secondChar):
        x1,y1,h1 = firstChar.brcX,firstChar.brcY,firstChar.brH
        x2,y2,h2 = secondChar.brcX,secondChar.brcY,secondChar.brH
        diff_pos = np.linalg.norm(np.array([x1,y1])-np.array([x2,y2]))
        diff_posy = abs(y1-y2)
        diff_size = abs(h1-h2)        
        return diff_pos,diff_posy,diff_size
    
    @staticmethod        
    def distanceBetweenChars(firstChar, secondChar):
        intX = abs(firstChar.brcX - secondChar.brcX)
        intY = abs(firstChar.brcY - secondChar.brcY)
    
        return math.sqrt((intX ** 2) + (intY ** 2))

    # use basic trigonometry (SOH CAH TOA) to calculate angle between chars
    @staticmethod
    def angleBetweenChars(firstChar, secondChar):
        fltAdj = float(abs(firstChar.brcX - secondChar.brcX))
        fltOpp = float(abs(firstChar.brcY - secondChar.brcY))
    
        if fltAdj != 0.0:                           # check to make sure we do not divide by zero if the center X positions are equal, float division by zero will cause a crash in Python
            fltAngleInRad = math.atan(fltOpp / fltAdj)      # if adjacent is not zero, calculate angle
        else:
            fltAngleInRad = 1.5708                          # if adjacent is zero, use this as the angle, this is to be consistent with the C++ version of this program
        # end if
    
        fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)       # calculate angle in degrees
    
        return fltAngleInDeg
    
    @staticmethod
    def findListOfListsOfMatchingChars(listOfPossibleChars):
                # with this function, we start off with all the possible chars in one big list
                # the purpose of this function is to re-arrange the one big list of chars into a list of lists of matching chars,
                # note that chars that are not found to be in a group of matches do not need to be considered further
        listOfListsOfMatchingChars = []                  # this will be the return value
    
        for possibleChar in listOfPossibleChars:                        # for each possible char in the one big list of chars
            listOfMatchingChars = Contour.findListOfMatchingChars(possibleChar, listOfPossibleChars)        # find all chars in the big list that match the current char
    
            listOfMatchingChars.append(possibleChar)                # also add the current char to current possible list of matching chars
    
            if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:     # if current possible list of matching chars is not long enough to constitute a possible plate
                continue                            # jump back to the top of the for loop and try again with next char, note that it's not necessary
                                                    # to save the list in any way since it did not have enough chars to be a possible plate
            # end if
    
                                                    # if we get here, the current list passed test as a "group" or "cluster" of matching chars
            listOfListsOfMatchingChars.append(listOfMatchingChars)      # so add to our list of lists of matching chars
    
            listOfPossibleCharsWithCurrentMatchesRemoved = []
    
                                                    # remove the current list of matching chars from the big list so we don't use those same chars twice,
                                                    # make sure to make a new big list for this since we don't want to change the original big list
            listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))
    
            recursiveListOfListsOfMatchingChars = Contour.findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      # recursive call
    
            for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        # for each list of matching chars found by recursive call
                listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             # add to our original list of lists of matching chars
            # end for
    
            break       # exit for
    
        # end for
    
        return listOfListsOfMatchingChars
    # end function
    
    ###################################################################################################
    @staticmethod
    def findListOfMatchingChars(possibleChar, listOfChars):
                # the purpose of this function is, given a possible char and a big list of possible chars,
                # find all chars in the big list that are a match for the single possible char, and return those matching chars as a list
        listOfMatchingChars = []                # this will be the return value
    
        for possibleMatchingChar in listOfChars:                # for each char in big list
            if possibleMatchingChar == possibleChar:    # if the char we attempting to find matches for is the exact same char as the char in the big list we are currently checking
                                                        # then we should not include it in the list of matches b/c that would end up double including the current char
                continue                                # so do not add to list of matches and jump back to top of for loop
            # end if
                        # compute stuff to see if chars are a match
            fltDistanceBetweenChars = Contour.distanceBetweenChars(possibleChar, possibleMatchingChar)
    
            fltAngleBetweenChars = Contour.angleBetweenChars(possibleChar, possibleMatchingChar)
    
            fltChangeInArea = float(abs(possibleMatchingChar.brA - possibleChar.brA)) / float(possibleChar.brA)
    
            fltChangeInWidth = float(abs(possibleMatchingChar.brW - possibleChar.brW)) / float(possibleChar.brW)
            fltChangeInHeight = float(abs(possibleMatchingChar.brH - possibleChar.brH)) / float(possibleChar.brH)
    
                    # check if chars match
            if (fltDistanceBetweenChars < (possibleChar.brdS * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
                fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
                fltChangeInArea < MAX_CHANGE_IN_AREA and
                fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
                fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):
    
                listOfMatchingChars.append(possibleMatchingChar)        # if the chars are a match, add the current char to list of matching chars
            # end if
        # end for
    
        return listOfMatchingChars                  # return result

import statistics
from ocr import TesserOCR
tesserocr = TesserOCR()

VIN_WIDTH_PADDING_FACTOR = 1.2
VIN_HEIGHT_PADDING_FACTOR = 1.17

EXPECTED_LENGTH = 17.0

SCALAR_RED = (0.0, 0.0, 255.0)

class String:
    def __init__(self):
        self.chars = []
        self.estimatedcharcount = 0
        self.charcount = 0
        self.charwidth = 0
        self.charheight = 0
        self.charleftmost = 0
        self.charrightmost = 0
        self.result = ""
        self.confidence = 0.0
        self.density = 0.0
        self.chargap = 0
        self.stringlength = 0
        self.StartIndex = -1
        self.FirstIndex = -1
        self.LastIndex = -1
        self.ROI = None
        self.segments = []
        self.meaningfulcharcount = 0
        self.isVIN = False
        
    def push(self,char):
        self.chars.append(char)
        self.charcount += 1
        
    def getitems(self):
        return self.chars
    
    def getlength(self):
        return self.charcount
    
    def sort(self):
        self.chars.sort(key=lambda char:char.brX)
        self.charleftmost=self.chars[0].brcX
        self.charrightmost=self.chars[self.charcount-1].brcX
        heights = []
        widths = []
        for char in self.chars:
            heights.append(char.brH)
            widths.append(char.brW)
        self.charwidth = statistics.median_high(widths)
        self.charheight = statistics.median_high(heights)
        self.setcenterline()
        self.setGap()
        self.calcConfidence()

    def tocontours(self):
        contours = []
        for char in self.chars:
            contours.append(char.contour)
        return contours
        
    def setcenterline(self):
        pts = []
        for char in self.chars:
            pts.append([char.brcX,char.brcY])
        [self.vx, self.vy, self.x, self.y] = cv2.fitLine(np.array(pts), cv2.DIST_HUBER, 0, 0.01, 0.01)
    
    def getcenterliney(self,x):
        return int(((x - self.x) * self.vy / self.vx) + self.y)

    def calcAngle(self):
        self.sort()#key = lambda char: char.brcX)        # sort chars from left to right based on x position    
        chars=self.chars
        FirstIndex = 0
        LastIndex = self.charcount - 1
        # calculate correction angle of plate region
        dist_y = chars[LastIndex].brcY - chars[FirstIndex].brcY
        dist = Contour.distanceBetweenChars(chars[FirstIndex], chars[LastIndex])
        radian = math.asin(dist_y / dist)
        angle = radian * (180.0 / math.pi)        
        self.angle = angle
        return angle
        #self.rmatrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)

    def setGap(self,gap=None):
        if gap is not None:
            self.chargap = gap
            return
        if self.charcount < 2:
            self.chargap = -1
            return
        gaps = []
        for i in range(self.charcount-1):
            dist = Contour.distanceBetweenChars(self.chars[i],self.chars[i+1])
            gaps.append(dist) 
        self.chargap = statistics.median_low(gaps)
        
    def calcConfidence(self):
        if self.charcount < 2:
            self.estimatedcharcount = 0
            return 0.0
        self.stringlength = Contour.distanceBetweenChars(self.chars[0], self.chars[self.charcount-1])
        self.estimatedcharcount = round(self.stringlength/self.chargap) + 1
        self.confidence = 1.0 - abs(self.estimatedcharcount - EXPECTED_LENGTH)/EXPECTED_LENGTH
        self.density = 1.0 - abs(self.charcount - EXPECTED_LENGTH)/EXPECTED_LENGTH
        if self.confidence < 0:
            self.confidence = 0.0
        self.setresult()
        return self.confidence

    def validateStartIndex(self):
        if (self.StartIndex - self.FirstIndex) < 2:
            return
        self.StartIndex = -1
    
    def ocrfiltering(self,isdebug=False):
        chars = []
        #self.calcangle()
        for i,candidate in enumerate(self.chars):
            ocrresult = candidate.char
            if ocrresult != u"@" and String.isdigist_or_english(ocrresult):
                #print ocrresult
                integer = ord(ocrresult)
                candidate.char = ocrresult
                self.meaningfulcharcount += 1
                chars.append(candidate)
                if integer == 76 and self.StartIndex == -1:
                    self.StartIndex = i
                if self.FirstIndex == -1:
                    self.FirstIndex = i
                self.LastIndex = i
        
        #
        self.validateStartIndex()
        #
        self.chars =  chars
        self.charcount = len(self.chars)
        #self.setGap()
        self.calcConfidence()

    @staticmethod
    def isdigist_or_english(txt):
        integer = ord(txt)
        #print ocrresult[0]
        if (48 <= integer and integer <= 57) or\
           (65 <= integer and integer <= 90) or\
           (97 <= integer and integer <= 122):
               return True
        return False
      
    @staticmethod
    def ocr1(chars):
        ocrresults = ""
        count = 0
        for candidate in chars:
            candidate.setBinary()
            binary = candidate.getBinary()
            #showResult("binary",binary)
            ocrresult = tesserocr.img2string_single(binary)
            if ocrresult != u"":
                candidate.char = ocrresult[0]
                if String.isdigist_or_english(ocrresult[0]):
                    count += 1
            else:
                candidate.char = "@"
            ocrresults += candidate.char
        return ocrresults,count

    @staticmethod
    def ocr2(img,chars):
        #https://stackoverflow.com/questions/9480013/image-processing-to-improve-tesseract-ocr-accuracy
        #https://github.com/tesseract-ocr/tesseract/wiki/ImproveQuality
        ocrresults = ""
        height,widht = img.shape
        count = 0
        for char in chars:
            x,y,w,h=char.brX,char.brY,char.brW,char.brH
            img_char = img[y:y+h,x:x+w]
            ratio = 20.0 / h
            resized = cv2.resize(img_char,None,fx=ratio,fy=ratio)
            h_,w_ = resized.shape
            w__ = w_/4*4+8
            h__ = h_/4*4+8
            temp = np.zeros((h__,w__),np.uint8)
            temp[2:2+h_,2:2+w_] = resized
            #showResult("temp",temp)
            ocrresult = tesserocr.img2string_single(temp)
            if ocrresult != u"":
                char.char = ocrresult[0]
                if String.isdigist_or_english(ocrresult[0]):
                    count += 1
            else:
                char.char = "@"
            ocrresults += char.char
        return ocrresults,count
                    
    def setresult(self):
        self.result = ""
        for char in self.chars:
            self.result += char.char
        return self.result

    @staticmethod
    def makesubstring(chars,StartIndex,endindex,gap=None,useLindex=True):
        string = String()
        endindex = endindex if endindex < len(chars) else (len(chars)-1)
        for i in range(StartIndex,endindex+1):
            string.push(chars[i])
        if useLindex:
            string.StartIndex = StartIndex
        string.LastIndex = endindex
        string.setGap(gap)
        string.calcConfidence()
        return string
    
    def refinestring(self):
        if self.StartIndex == -1:
            return
        chars = self.chars
        while ord(chars[0].char) != 76:
            del chars[0]
            self.charcount -= 1
        self.chars = chars
        self.setresult()
        self.calcConfidence()

    def pop(self):
        if len(self.chars) == 0:
            return
        chars = self.chars
        tmp = chars[len(chars)-1]
        del chars[len(chars)-1]
        self.charcount -= 1
        self.chars = chars
        return tmp

    def isAcceptable(self):
        #print self.confidence, self.density, self.meaningfulcharcount
        if self.confidence < 0.9 or self.density < 0.9 or self.meaningfulcharcount < 4:
            return False
        self.isVIN = True
        return True
    
    @staticmethod
    def filtering_by_ocr(strings):
        strings_ = []
        for string in strings:
            chars = string.chars
            ocrresults,count = String.ocr1(chars)
            string.ocrfiltering()
            print string.confidence,string.result,string.estimatedcharcount
            if "L" in string.result and string.StartIndex != -1:
                string = String.makesubstring(chars,string.StartIndex,string.LastIndex,string.chargap)#string.refinestring()#
                string.meaningfulcharcount = count
                confidence = string.confidence
                #print confidence,string.result,string.estimatedcharcount
                print confidence,string.result.encode('utf-8').strip(),string.estimatedcharcount
                if confidence > 0.95:
                    strings_.append(string)
                    break
                elif string.charcount > 4 and string.estimatedcharcount < EXPECTED_LENGTH:
                    while string.confidence < 1.0 and string.LastIndex < (len(chars)-1):
                        #tmp = string.pop():
                        string.push(chars[string.LastIndex+1])
                        string.calcConfidence()
                        string.LastIndex+=1
                        print string.confidence,string.result.encode('utf-8').strip(),string.charcount

                        if string.confidence < confidence:
                            string.pop()
                            string.calcConfidence()
                            string.LastIndex-=1
                            print string.confidence,string.result
                            break
                        confidence = string.confidence

                        #print string.confidence
                    strings_.append(string)
                    if string.confidence > 0.9:
                        break
                elif string.estimatedcharcount > EXPECTED_LENGTH:
                    while string.confidence < 1.0:
                        tmp = string.pop()
                        string.calcConfidence()
                        string.LastIndex-=1

                        if string.confidence < confidence:
                            string.push(tmp)
                            string.calcConfidence()
                            string.LastIndex+=1
                            break
                        confidence = string.confidence

                        #print string.confidence
                    strings_.append(string)
                    if string.confidence > 0.9:
                        break
            else:#elif string.confidence > 0.7:
                string = String.makesubstring(chars,string.FirstIndex,string.LastIndex,string.chargap,False)
                string.meaningfulcharcount = count
                strings_.append(string)
        return strings_

    @staticmethod
    def filtering_by_criteria(strings):
        strings_ = []
        for string in strings:
            if string.confidence > 0.85:#if string.isAcceptable():
                strings_.append(string)
        return strings_
    
    @staticmethod
    def sortall(strings):
        for string in strings:
            string.sort()

    @staticmethod
    def makesegments(image,string):
        string.ROI = String.extractROI(image,string,False)
        bgr = string.ROI.cropped
        chars =  string.chars
        FirstIndex = 0
        LastIndex = string.charcount - 1
        h,w = bgr.shape[:2]
        delta = w - (chars[LastIndex].brX + chars[LastIndex].brW - chars[FirstIndex].brX)
        first = chars[0].brX
        for char in chars:
            x = char.brX - first + delta
            segment = bgr[1:string.charheight,x:x+string.charwidth,:]
            string.segments.append(segment)
            showResult("segment",segment)
        
    @staticmethod
    def extractROI(img, string,vinmode=True):
        roi = ROI()           # this will be the return value    
        string.sort()#key = lambda char: char.brcX)        # sort chars from left to right based on x position    
        chars=string.chars
        FirstIndex = 0
        LastIndex = string.charcount - 1
        # calculate correction angle of plate region
        angle = string.calcAngle()
        # calculate the center point of the plate
        center_x = (chars[FirstIndex].brcX + chars[LastIndex].brcX) / 2.0
        center_y = (chars[FirstIndex].brcY + chars[LastIndex].brcY) / 2.0
        stringwidth = (chars[LastIndex].brX + chars[LastIndex].brW - chars[FirstIndex].brX)
        stringheight = int(string.charheight * VIN_HEIGHT_PADDING_FACTOR)  
        
        if vinmode:
            # calculate plate width and height
            stringwidth *= (1 + 0.06*(EXPECTED_LENGTH - string.estimatedcharcount) + 0.06)
            stringwidth = int(stringwidth)
            
            if string.StartIndex == -1 and string.confidence < 1:
                gapx = string.chargap/ 2 
                gapy = gapx * math.sin((string.angle/180.0) * math.pi)
                center_x -= gapx
                center_y += gapy
            else:
                gapx = (stringwidth - string.chargap) / 2 
                gapy = gapx * math.sin((string.angle/180.0) * math.pi)
                center_x = chars[FirstIndex].brcX + gapx
                center_y = chars[FirstIndex].brcY + gapy
        else:
            stringwidth *= (1 + 0.06*(EXPECTED_LENGTH - string.estimatedcharcount) + 0.06)
            stringwidth = int(stringwidth)
            
        center = center_x, center_y
        # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
        roi.rrLocationOfPlateInScene = (tuple(center), (stringwidth, stringheight), angle )  
        # final steps are to perform the actual rotation
        # get the rotation matrix for our calculated correction angle
        rotationMatrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
        h, w, c = img.shape      # unpack original image width and height
        rotated = cv2.warpAffine(img, rotationMatrix, (w, h))       # rotate the entire image
        cropped = cv2.getRectSubPix(rotated, (stringwidth, stringheight), tuple(center))
        roi.cropped = cropped         # copy the cropped plate image into the applicable member variable of the possible plate
        
        return roi

    @staticmethod
    def string2bbox(img,string):
        ROI = String.extractROI(img, string)
        return cv2.boxPoints(ROI.rrLocationOfPlateInScene)       
        
    @staticmethod
    def markimage(img,string):
        ROI = String.extractROI(img, string)
        p2fRectPoints = cv2.boxPoints(ROI.rrLocationOfPlateInScene)
        cv2.line(img, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 1)
        cv2.line(img, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 1)
        cv2.line(img, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 1)
        cv2.line(img, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 1)
        return img
        
    @staticmethod
    def mark(img,strings,isdebug=False):
        import types
        if type(strings) is types.ListType:
            for string in strings:
                String.markimage(img,string)
        else:
            String.markimage(img,strings)
        #showResult("roi",ROI.imgPlate)
        #showResult("marked",img)
        return img

class ROI:

    # constructor #################################################################################
    def __init__(self):
        self.cropped = None
        self.gray = None
        self.thr = None

        self.rrLocationOfPlateInScene = None

        self.strChars = ""

