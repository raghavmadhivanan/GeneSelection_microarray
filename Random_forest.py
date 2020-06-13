# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 23:21:07 2020

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
    data = pd.DataFrame(data)                                        #X.iloc[116,43072]                 # in glioma data this index contains nan value
    X = data.iloc[:,0:-1]                      #Extracting the features and labels
    y = data.iloc[:,-1].values.reshape(-1,1)    # reshaping the series 
    
    from sklearn.model_selection import train_test_split      # splitting the dataset in traing and testset 
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42)   # random state for reproducing the results
    # =============================================================================
    # 
    # Rabdom forest classifier  #### Classifier Ensemble of decision tree - use as base estimator
    # =============================================================================
    from sklearn.ensemble import RandomForestClassifier
    # Instantiate a random forests regressor 'rf' 400 estimators
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=0.12, random_state=1)
    # Fit 'rf' to the training set
    rf.fit(X_train, y_train)
    
    
    
    # =============================================================================
    # test the data$
    # =============================================================================
    
    # Predict the test set labels 'y_pred'
    y_pred = rf.predict(X_test)
    
    # =============================================================================
    # checking the accuracy
    # =============================================================================
    
    from sklearn import metrics
    print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
    accuracy.append(metrics.accuracy_score(y_test,y_pred))

    
    # Create a pd.Series of features importances
    importances_rf = pd.Series(rf.feature_importances_, index = X.columns)
    # Sort importances_rf
    sorted_importances_rf = importances_rf.sort_values()
    # Make a horizontal bar plot
    importances_rf.nlargest(10).plot(kind='barh')           # for 10 largest values i.e most important 10 feaatures for gene selection
    plt.ylabel("Top 10 Significant genes")
    plt.title(j)
    plt.show()
