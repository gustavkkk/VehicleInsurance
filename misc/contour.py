# self.py

import cv2
import numpy as np
import math

from feature.bbox import contour2fitline,contour2bbox
# Char Contour Filtering Definition
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

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
    def __init__(self, _contour, image=None):
        
        if image is not None:
            self.setImageSize(image)
        else:
            self.image_height,self.image_width = 0,0
        
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
        cv2.drawContours(self.binary, [contour_], -1, (255,255,255),-1)

    def getBinary(self):
        return self.binary
        
    def setImageSize(self,image):
        self.image_height,self.image_width = image.shape[:2]
        
    def checkIfPossibleChar(self):
        # this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
        # note that we are not (yet) comparing the char to other chars to look for a group
        if (self.brA > MIN_PIXEL_AREA and
            self.brW > MIN_PIXEL_WIDTH and self.brH > MIN_PIXEL_HEIGHT and
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

class String:
    def __init__(self):
        self.chars = []
        self.length = 0
    def add(self,char):
        self.chars.append(char)
        self.length += 1
    def getitems(self):
        return self.chars
    def getlength(self):
        return self.length
        
class ROI:

    # constructor #################################################################################
    def __init__(self):
        self.imgPlate = None
        self.imgGrayscale = None
        self.imgThresh = None

        self.rrLocationOfPlateInScene = None

        self.strChars = ""

