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
from image_processing import *
from key_check import key_check

data_1 = np.load("training_data_100x80_willow_springs_raceway_new.npy")
data_2 = np.load("training_data_100x80_willow_springs_raceway_rev_new.npy")

data = np.append(data_1, data_2, axis=0)

# stacking each img array
X = np.stack(data[:,0], axis=0)
X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
y_temp = np.stack(data[:,1], axis=0)

# only steering data
y_left = y_temp[:,1]
y_right = y_temp[:,3]
y_straight = np.logical_not(np.logical_xor(y_left,y_right))

# collecting where the particular pos is 1 and taking first 4000 indexes
# check smallest sum first > np.sum(y_left==True) | 4000 for old, 3400 for new
y_left_act = np.where(y_left==True)[0][:3400]
y_right_act = np.where(y_right==True)[0][:3400]
y_straight_act = np.where(y_straight==True)[0][:3400]

# taking only these selected index data
X_temp = np.copy(X)
selected_indexes = np.append(np.append(y_left_act,y_right_act),y_straight_act)
X = X[selected_indexes]

# for ann flattening array
data_X = np.zeros((len(X), X[0].shape[0]*X[0].shape[1]))
for i in range(len(X)):
    data_X[i] = X[i].reshape((1,X[i].shape[0]*X[i].shape[1]))

y_left = y_temp[selected_indexes,1]
y_right = y_temp[selected_indexes,3]
y_straight = np.logical_not(np.logical_xor(y_left,y_right))
y = np.stack([y_left,y_right,y_straight],axis=1)

#for i in range(len(data)):
#    X[i] = data[i][0].reshape(-1)
#    y[i] = np.array(data[i][1])
    

from sklearn.model_selection import train_test_split
# X for cnn / data_X for ann
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)
#
import pickle
pik_file = open("classifier_cat_scaler_new.pickle","wb")
pickle.dump(scaler, pik_file)
pik_file.close()

import pickle
with open("classifier_cat_scaler.pickle","rb") as fl:
    scaler = pickle.load(fl)
with open("classifier_pc_100x80.pickle","rb") as fl:
    classifier = pickle.load(fl)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.callbacks import EarlyStopping
from min_pooling import MinPooling2D

classifier = Sequential()

# adding conv layer
classifier.add(Conv2D(32, (5, 5), input_shape=(80, 100, 1)))
classifier.add(MaxPooling2D(pool_size=(2,2)))
## adding 2nd conv. layer
classifier.add(Conv2D(64, (3,3)))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#
classifier.add(Flatten())
# adding full connection ## remove input_shape for CNN
#classifier.add(Dense(4048, activation='sigmoid', input_shape=(8000,)))
classifier.add(Dense(512, activation='sigmoid'))
classifier.add(Dense(128, activation='sigmoid'))
classifier.add(Dense(32, activation='sigmoid'))

#classifier.add(Dense(512, activation='sigmoid'))
classifier.add(Dense(3, activation='softmax'))

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) 

early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, 
                              mode='min', restore_best_weights=True)

his = classifier.fit(X_train, y_train, batch_size=50, epochs=15, 
                     validation_data=(X_test, y_test), callbacks=[early_stopper])

y_pred = classifier.predict(X_test)
y_pred_t = np.copy(y_pred)
y_pred_t[np.arange(len(y_pred)),y_pred.argmax(axis=1)]=1
y_pred_t[y_pred_t!=1]=0

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.argmax(y_test,axis=1),np.argmax(y_pred_t,axis=1))

import pickle

classifier_cat_file = open("classifier_pc_100x80_new.pickle","wb")
pickle.dump(classifier, classifier_cat_file)
classifier_cat_file.close()

###############

pik_file = open("classifier_bin.pickle","rb")
classifier = pickle.load(pik_file)
pik_file.close()

y_pred = classifier.predict(X_test)

res_y_pred = np.copy(y_pred)
res_y_pred[res_y_pred>=0.5] = 1
res_y_pred[res_y_pred<0.5] = 0

from sklearn.metrics import confusion_matrix
cm_0 = confusion_matrix(y_test[:,0],res_y_pred[:,0])
cm_1 = confusion_matrix(y_test[:,1],res_y_pred[:,1])
cm_2 = confusion_matrix(y_test[:,2],res_y_pred[:,2])
cm_3 = confusion_matrix(y_test[:,3],res_y_pred[:,3])
