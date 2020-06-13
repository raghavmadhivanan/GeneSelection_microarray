# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 00:21:43 2020

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
        
files.remove("BreastCancer4_Disease.csv")            # adaboost is working for multi class classificatio for we can use samme and samme.r
files.remove("Crohn_Disease.csv")
files.remove("Leukemia_Disease.csv")
files.remove("Leukemia3_Disease.csv")
files.remove("Mislaneous_Disease.csv")
files.remove("Mislenious2_Disease.csv")
files.remove("SRBCT_Disease.csv")
files.remove("Glioma_Disease.csv")
files.remove("Sarcoma_Disease.csv")


accuracy = []
scor = []
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
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    
    # Instantiate a classification-tree 'dt'
    dt = DecisionTreeClassifier(max_depth=1, random_state=1)
    # Instantiate an AdaBoost classifier 'adab_clf'
    adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)
    # Fit 'adb_clf' to the training set 
    adb_clf.fit(X_train, y_train)
    # Predict the test set probabilities of positive class
    y_pred_proba = adb_clf.predict_proba(X_test)[:,1]
    y_pred  = adb_clf.predict(X_test)

    
    from sklearn.metrics import roc_auc_score
    from sklearn import metrics
    
    # Evaluate test-set roc_auc_score
    adb_clf_roc_auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Print adb_clf_roc_auc_score
    print('ROC AUC score: {:.2f}'.format(adb_clf_roc_auc_score))
    print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
    accuracy.append(metrics.accuracy_score(y_test,y_pred))
    scor.append(adb_clf_roc_auc_score)
    
    from sklearn.metrics import roc_curve  
    fper, tper, thresholds = roc_curve(y_test, y_pred_proba) 
    plt.plot(fper, tper)
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Adaboost ROC curve')
    # show the plot
    plt.show()