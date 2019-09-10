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
from image_processing import intensity_slice
from key_check import key_check
from directkeys import PressKey, ReleaseKey, W, A, S, D
import pickle

pik_file = open("classifier_bin.pickle","rb")
classifier = pickle.load(pik_file)
pik_file.close()

pik_file = open("classifier_bin_scaler.pickle","rb")
scaler = pickle.load(pik_file)
pik_file.close()

from collect_train_data import process_image
    
def set_frame():
    print("set frame..")
    while True:
        test_img = np.array(ImageGrab.grab(bbox=(5,30,1280,745)))
        cv2.imshow('window',test_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
def start():
    
    set_frame()
    input("press enter to start:")
    time.sleep(5)
    last_time = time.time()
    delay = []
    while True:
        
        last_time = time.time()
        printscreen =  np.array(ImageGrab.grab(bbox=(5,30,1280,745)))
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
        
        X_input = printscreen.reshape((1,printscreen.shape[0]*printscreen.shape[1]))
        X_input = scaler.transform(X_input)
        y_output = classifier.predict(X_input)[0]
        
        delay.append(time.time()-last_time)
        print('loop took {:.4f} seconds,'.format(delay[-1]),end="")
        # pressing keys
        # if S is activated else if W is activated
#        if y_output[2] >= 0.5:
#            PressKey(S)
#            print(" S ",end="")
#        elif y_output[0] >= 0.5:
#            PressKey(W)
#            print(" W ",end="")
        # if A is activated else if D is activated
        if y_output[1] >= 0.5:
            PressKey(A)
            print(" A ",end="")
        elif y_output[3] >= 0.5:
            PressKey(D)
            print(" D ",end="")
        # delay for a bit
#        time.sleep(0.1)
        # releasing keys
        # if S is activated else if W is activated
#        if y_output[2] < 0.5:
#            ReleaseKey(S)
#        if y_output[0] < 0.5:
#            ReleaseKey(W)
        # if A is activated else if D is activated
        if y_output[1] < 0.5:
            ReleaseKey(A)
        if y_output[3] < 0.5:
            ReleaseKey(D)
        
        print()