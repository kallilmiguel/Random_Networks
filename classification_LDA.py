#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 11:01:17 2022

@author: kallil
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import os

DATASET = "USPTex"
DIR = 'data/'+DATASET+'/features/'

path_list = [DIR+paths for paths in os.listdir(DIR)]

path_list = sorted(path_list, key=str.lower)

y = np.loadtxt(DIR+"classes.csv")


counter=1
global_accuracies = []
for p in path_list:
    if("classes.csv" not in p and ".DS" not in p):       
        print(f"Configuração {counter} de {len(path_list)} em treinamento -- {p}")
        counter+=1
        X = np.genfromtxt(p, delimiter=',')
        
        
        loo = LeaveOneOut()
        loo.get_n_splits(X)
        
        config_accuracies = []
        
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]  
   
            std = StandardScaler()
            X_train = std.fit_transform(X_train)
            X_test = std.transform(X_test)
            
            #kernel SVM
            # clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
            # clf.fit(X_train,y_train)
            # y_pred = clf.predict(X_test)
            
            #LDA Classifier
            lda = LDA(solver='svd')
            lda.fit(X_train, y_train)
            y_pred = lda.predict(X_test)
            
            
            config_accuracies.append(accuracy_score(y_test, y_pred))
            
        config_accuracies = np.array(config_accuracies)
        global_accuracies.append(np.mean(config_accuracies))

global_accuracies = np.array(global_accuracies)
np.savetxt(DIR+"accuracies.csv", global_accuracies, delimiter=',')

