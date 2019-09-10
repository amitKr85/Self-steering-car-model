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

img = cv2.resize(cv2.imread('lykan_1920x1080.jpg'),(1366,768))
cv2.imshow('img',img)
kernel = np.ones((3,3),np.float32)*-1
kernel[1,1] = 8
kernel /= 9
filt_img = (img.astype(int) + cv2.filter2D(img,-1,kernel).astype(int)).astype('uint8')
cv2.imshow('filtered',filt_img)
