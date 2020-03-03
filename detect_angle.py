#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 08:10:43 2017

@author: junying
"""
from misc.switch import switch

def detect_angle(image,
                 bbox_lp,
                 bbox_car):
    height_screen,width_screen,channels = image.shape
    x_center = int(width_screen/2)
    #y_center = int(h/2)
    #
    print bbox_car
    print bbox_car[:,0]
    print bbox_car[:,1]
    xmin_car = bbox_car[:,0].min()
    ymin_car = bbox_car[:,1].min()
    xmax_car = bbox_car[:,0].max()
    ymax_car = bbox_car[:,1].max()
    width_car = xmax_car - xmin_car
    height_car = ymax_car - ymin_car
    xmin_lp = bbox_lp[:,0].min()
    ymin_lp = bbox_lp[:,1].min()
    xmax_lp = bbox_lp[:,0].max()
    ymax_lp = bbox_lp[:,1].max()
    width_lp = xmax_lp -xmin_lp
    height_lp = ymax_lp - ymin_lp
    # check if car is in the left of the screen or not
    result = "不通过"
    report = []
    if xmax_car < x_center and xmin_car < x_center:
        report.append(r"车 : 左")
        return result,report
    elif xmax_car > x_center and xmin_car > x_center:
        report.append(r"车 : 右")
        return result,report
    else:
        report.append(r"车 : 中央")
    # check vehicle size
    if (width_car*height_car)*2.7 < (width_screen*height_screen) \
        and (height_screen - ymax_car) * 2.2 > (ymax_car - ymin_car):
        report.append(r"车 : 过远")
        #return result,report
    elif (width_car*2.1 < width_screen):
        report.append(r"车 : 过远,角度不够")
        #return result,report
    elif (height_car*2.1 < height_screen):
        report.append(r"车 : 不全面-上")
        #return result,report
    # check if license plate is in the left of the screen or not
    if (xmax_lp < x_center and xmin_lp < x_center):
        lp_left = True
        lp_to_end = xmin_lp - xmin_car
        lp_to_center = x_center - xmax_lp#
        x_corner = xmax_lp + width_lp*1.5#between car side and car front
        report.append(r"车牌 : 左前,右后")
    elif (xmax_lp > x_center and xmin_lp > x_center):
        lp_left = False
        lp_to_end = xmax_car - xmax_lp
        lp_to_center =  xmin_lp - x_center#
        x_corner = xmin_lp - width_lp*1.5
        report.append(r"车牌 : 右前,左后")
    else:
        report.append(r"车牌 : 中央")
        return result,report# LP located in the center of the screen
    # check vehicle position
    if xmin_car > 10*(width_screen - xmax_car) \
        and lp_to_end *10 < width_lp:
        report.append(r"车 : 不全面-左")
        return result,report
    if xmin_car*10 < (width_screen - xmax_car)\
        and lp_to_end *10 < width_lp:
        report.append(r"车 : 不全面-右")
        return result,report
    # check angle with left-cetner:center-right
    if (x_corner - xmin_car)*1.5 < (xmax_car - x_corner):
        if lp_left:
            report.append(r"角度过度")
        else:
            report.append(r"角度不够")
        return result,report
    elif (x_corner - xmin_car) > 2.5 * (xmax_car - x_corner):
        if lp_left:
            report.append(r"角度不够")
        else:
            report.append(r"角度过度")
        return result,report
    else:
        report.append("角度正确")
    return "通过",report
    
            
    
    
    