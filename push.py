# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 15:12:09 2017

@author: Frank
"""
def push(comment):
    #import subprocess as sub
    #sub.Popen('cmd /K dir')
    #sub.Popen(['cmd', '/K', 'dir'])
    
    #import pyautogui
    #pyautogui.typewrite('git add .')
    #pyautogui.keyDown('enter')
    #pyautogui.typewrite('git commit -m %s'%comment)
    #pyautogui.keyDown('enter')
    #pyautogui.typewrite('git push origin master')
    #pyautogui.keyDown('enter')
    
    import os
    os.system('git add .')
    os.system('git commit -m %s'%comment)
    os.system('git push origin master')

import sys

def main():
    if len(sys.argv) < 2:
        print('leave comment please.')
        comment = input()
        push(comment)
    else:
        push(sys.argv[1])
        
if __name__ == "__main__":
    main()