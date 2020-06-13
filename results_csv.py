# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:11:10 2020

@author: user
"""


import csv  
    
# field names  
fields = ['Dataset', 'Adaboostclassifier_Auc_roc_score']  
    
# data rows of csv file  
rows = [ files,  
        scor]  
    
# name of csv file  
filename = "AdabboostClassifierResults.csv"
    
# writing to csv file  
with open(filename, 'w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
        
    # writing the fields  
    csvwriter.writerow(fields)  
        
    # writing the data rows  
    csvwriter.writerows(rows) 