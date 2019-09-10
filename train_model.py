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

data_1 = np.load("training_data.npy")
data_2 = np.load("training_data_2.npy")
data_3 = np.load("training_data_3.npy")
data_4 = np.load("training_data_4.npy")

data = np.append(np.append(np.append(data_1, data_2, axis=0), data_3, axis=0), data_4, axis=0)

# flattening each img array
X = np.zeros((len(data),len(data[0][0].reshape(-1))))
y = np.zeros((len(data),len(data[0][1])))
for i in range(len(data)):
    X[i] = data[i][0].reshape(-1)
    y[i] = np.array(data[i][1])
    

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


pik_file = open("classifier_bin_scaler.pickle","wb")
pickle.dump(scaler, pik_file)
pik_file.close()

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(4552,kernel_initializer='uniform',activation='sigmoid',input_shape=(9100,)))

classifier.add(Dense(4,kernel_initializer='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=50, epochs=20)

import pickle

classifier_cat_file = open("classifier_bin.pickle","wb")
pickle.dump(classifier, classifier_cat_file)
classifier_cat_file.close()

###############

pik_file = open("classifier_bin.pickle","rb")
classifier = pickle.load(pik_file)
pik_file.close()
###########################

y_pred = classifier.predict(X_test)

res_y_pred = np.copy(y_pred)
res_y_pred[res_y_pred>=0.5] = 1
res_y_pred[res_y_pred<0.5] = 0

from sklearn.metrics import confusion_matrix
cm_0 = confusion_matrix(y_test[:,0],res_y_pred[:,0])
cm_1 = confusion_matrix(y_test[:,1],res_y_pred[:,1])
cm_2 = confusion_matrix(y_test[:,2],res_y_pred[:,2])
cm_3 = confusion_matrix(y_test[:,3],res_y_pred[:,3])
