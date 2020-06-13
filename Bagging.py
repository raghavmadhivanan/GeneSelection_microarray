# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 00:37:22 2020

@author: user
"""


import pandas as pd                                      #importing the library
import numpy as np
import matplotlib.pyplot as plt

import os
files = []
for i in os.listdir("main_data"):
    if i.endswith(".csv"):
        files.append((i))
accuracy = []
for j in files:
    data = pd.read_csv(j)            #Loading the dataset
    
    #Exploratory data anlysis
    
    type(data)                    # type of dataFrame
    
    # =============================================================================
    # #preprocessing
    # =============================================================================
    
    data = data.drop(["Unnamed: 0"],axis =1)      # Removing the Unwanted column 
    data.rename(index={-1:'y'},inplace = True)

    from sklearn.preprocessing import LabelEncoder   # label encoding the labels
    labelencoder = LabelEncoder()
    data['y'] = labelencoder.fit_transform(data['y'])
    
    
    np.where(np.isnan(data))
    data = np.nan_to_num(data)
    data = pd.DataFrame(data)
    
    X = data.iloc[:,0:-1]                      #Extracting the features and labels
    y = data.iloc[:,-1].values.reshape(-1,1)    # reshaping the series 
    
    data.isnull().sum()
    
    from sklearn.model_selection import train_test_split      # splitting the dataset in traing and testset 
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42)
       # random state for reproducing the results
       
       
    # Import models and utility functions
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    
    # Instantiate a classification-tree 'dt'
    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=1)
    
    # Instantiate a BaggingClassifier 'bc'
    bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1) # n_jobs=-1 means that all the CPU cores are used in computation. 
    
    # Fit 'bc' to the training set
    bc.fit(X_train, y_train)
    
    # Predict test set labels
    y_pred = bc.predict(X_test)
    
    
    
    from sklearn import metrics
    
    # Evaluate and print test-set accuracy
    
    print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
    accuracy.append(metrics.accuracy_score(y_test,y_pred))

    