# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:07:43 2019

@author: AMIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from image_processing import *
import cv2
import time


def draw_lines(img, lines):
    for line in lines:
        coords = line[0]
        cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)


def roi(img, vertices):
    #blank mask:
    mask = np.zeros_like(img)
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked

def process_image(img):
    res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    res = intensity_slice(res, 40, 120, reverse=True)
#    res = cv2.blur(res,(10,10))
#    res = cv2.Canny(res, 100, 150)
#    vertices = np.array([[0,600],[0,200],[200,150],[600,150],[800,200],[800,600],[750,600],[400,500],[50,600]], np.int32)
#    res = roi(res, [vertices])
#    res = cv2.GaussianBlur(res, (5,5), 0)
#    # img, p, 0, minVote, minLength, maxGap
#    lines = cv2.HoughLinesP(res, 1, np.pi/180, 180, 20, 15)
#    draw_lines(res, lines)
#    
#    res = cv2.resize(res,(400,310))
    return res


def show_imgs(img_list, r, c, fig_num,size=(16,20)):    
    plt.figure(fig_num,figsize=size)
    for i in range(r):
        for j in range(c):
            if i*c+j >= len(img_list):
                break
            plt.subplot(r,c,i*c+j+1)
            # plt.imshow(img_list[i*c+j],cmap='gray')
            plt.imshow(cv2.cvtColor(img_list[i*c+j],cv2.COLOR_BGR2RGB))
            plt.title(i*c+j)
    plt.show()



last_time = time.time()


arr = np.load("img_color_data_800x600.npy")

#show_imgs(arr[40:],5,4,0)
imgs = arr[[3,4,8,11,12,22,26,28,34,40,44,45,46,48,51,53]]
show_imgs(imgs,4,4,0)
f_4 = imgs[[3,5,7,13]]
show_imgs(f_4,2,2,0)

def retain_grayscale_BGR(img,lim):
    
#    img = np.array(img,copy=True,dtype='int')
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    
#    lim = 10
    r_g_diff = np.abs(r-g)<lim
    g_b_diff = np.abs(g-b)<lim
    r_b_diff = np.abs(r-b)<lim
    
    mask = r_g_diff & g_b_diff & r_b_diff
    
    r = r*mask
    g = g*mask
    b = b*mask
    
    return np.stack([b,g,r],axis=2)
    
def process_img(img,lim):
    res = retain_grayscale_BGR(img,lim)
    res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    res = intensity_slice(res ,40,120)
#    res = cv2.erode(res, np.ones((3,3)), iterations=1)
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, np.ones((9,9)))
#    
#    res = cv2.blur(res,(10,10))
    return res
    
f_4_10p = []
f_4_15p = []
f_4_20p = []
for img in imgs:
#    f_4_p.append(intensity_slice_RGB(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),(40,110),(40,120),(40,120),True))
#    f_4_10p.append(process_img(img,10))
    f_4_15p.append(process_img(img,15))
#    f_4_20p.append(process_img(img,20))
    
#show_imgs(f_4_10p,2,2,1)
show_imgs(f_4_15p,4,4,21)
#show_imgs(f_4_20p,2,2,3)

#f4p=[]
#for i in range(len(f_4_15p)):
#    f4p.append(intensity_slice(f_4_15p[i],40,120))
#
##for i in range(len(f_4)):
##    f_4_p[i] = cv2.Canny(f_4_p[i], 350, 400)
#
#show_imgs(f4p, 2, 2, 0)

#res50 = intensity_slice(imgs, 50, 100, reverse=True)
#show_imgs(res50, 1)
#for i in range(len(res50)):
#    res50[i] = cv2.blur(res50[i],(10,10))
#show_imgs(res50, 2)
#res50_siz = np.zeros((len(imgs),60,130))
#for i in range(len(res50)):
#    res50_siz[i] = cv2.resize(res50[i],(130,60))
#show_imgs(res50_siz, 3, size=None)

print("total time taken:",time.time()-last_time,"secs")