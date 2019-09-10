# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:07:43 2019

@author: AMIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import ImageGrab
import time
from image_processing import intensity_slice,retain_grayscale_BGR
from key_check import key_check
from directkeys import PressKey, ReleaseKey, W, A, S, D
import pickle

with open("classifier_cat_scaler_new.pickle","rb") as fl:
    scaler = pickle.load(fl)
with open("classifier_pc_100x80_new.pickle","rb") as fl:
    classifier = pickle.load(fl)

from collect_train_data import set_frame
from collect_train_data import process_image
#
#def process_image(img,lim=15):
#    res = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#    res = retain_grayscale_BGR(res,lim)
#    res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
#    res = intensity_slice(res ,40,120)
##    res = cv2.erode(res, np.ones((3,3)), iterations=1)
#    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, np.ones((9,9)))
##    
##    res = cv2.blur(res,(10,10))
#    res = cv2.resize(res,(100,80))
#    return res

#def set_frame():
#    print("set frame..")
#    while True:
#        test_img = np.array(ImageGrab.grab(bbox=(5,30,800,625)))
#        cv2.imshow('window',test_img)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break
#    
def start():
    
    set_frame()
    input("press enter to start:")
    time.sleep(5)
    last_time = time.time()
    delay = []
    
    left_time = 0
    right_time = 0
    
    while True:
        
        last_time = time.time()
        printscreen =  np.array(ImageGrab.grab(bbox=(5,30,800,625)))
        printscreen = process_image(printscreen)
        cv2.imshow('window',printscreen)
        ch = cv2.waitKey(1) & 0xFF
        if  ch == ord('q'):
            cv2.destroyAllWindows()
            print(np.average(delay))
            break
        if ch == ord('o'):
            input("press enter to continue:")
            time.sleep(5)
            continue
        
#        X_input = printscreen.reshape((1,printscreen.shape[0],printscreen.shape[1],1))
        X_input = printscreen.reshape((1,printscreen.shape[0]*printscreen.shape[1]))
        X_input = scaler.transform(X_input)
        y_output = classifier.predict(X_input)[0]
        y_i = y_output.argmax()
        
        delay.append(time.time()-last_time)
        print('loop took {:.4f} seconds, conf={} '.format(delay[-1],y_output[y_i]),end="")
        # pressing keys
        # if S is activated else if W is activated
#        if y_output[2] >= 0.5:
#            PressKey(S)
#            print(" S ",end="")
#        elif y_output[0] >= 0.5:
#            PressKey(W)
#            print(" W ",end="")
        # if A is activated else if D is activated
#        if y_output[1] >= 0.5:
#            PressKey(A)
#            print(" A ",end="")
#        elif y_output[3] >= 0.5:
#            PressKey(D)
#            print(" D ",end="")
        
        if y_i == 0:
            print(" A ",end="")
            PressKey(A)
        if y_i == 1:
            print(" D ",end="")
            PressKey(D)
        
#        time.sleep(0.5*(y_output[y_i]))
        # delay for a bit
        time.sleep(0.2)
        # releasing keys
        # if S is activated else if W is activated
#        if y_output[2] < 0.5:
#            ReleaseKey(S)
#        if y_output[0] < 0.5:
#            ReleaseKey(W)
        # if A is activated else if D is activated
#        if y_output[1] < 0.5:
#            ReleaseKey(A)
#        if y_output[3] < 0.5:
#            ReleaseKey(D)
        if y_i == 0:
            ReleaseKey(A)
        if y_i == 1:
            ReleaseKey(D)
        
        print()