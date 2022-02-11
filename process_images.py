# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 10:38:21 2022

@author: kalli
"""

import ctypes
import numpy as np
import os
import cv2
import cython

DATABASE = "USPTex"
DATADIR = "data/"+DATABASE+"/images/"
SAVEDIR = "data/"+DATABASE+"/matrices/"


path_list = [imagepath for imagepath in os.listdir(DATADIR)]

for p in path_list:
    
    img = cv2.imread(DATADIR+p, cv2.IMREAD_GRAYSCALE)
    if(p[-4:] == ".png"):
        with open(SAVEDIR+p[:-4]+'.txt', 'w') as f:
            img = img.flatten()
            f.write(str(len(img))+'\n')
            np.savetxt(f, img, fmt='%d')
            f.close()
