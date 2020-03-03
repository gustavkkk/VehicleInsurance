#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 22:27:31 2017

@author: junying
"""
import re

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