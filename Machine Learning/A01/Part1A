# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:14:50 2021

@author: reddy
"""

import numpy as np
import csv
from collections import Counter

'''The minkowski distance function will take feature data,query instance 
    and hyper parameter 'a' as input and returns the distance'''
def minkowski_distance(feature_data, query_inst, a):
    # Using numpy broadcasting to solve the Minkowski distance formula
    res = (np.abs(feature_data - query_inst)**a).sum(axis=1)**(1/a)
    return res
    
# predict class will predict the class lables for the test data and returns the predicted lables.
def predict_class(feature_data, class_lables, query_inst, a, k):
    dists = minkowski_distance(feature_data,query_inst,a) # calling the minkowski distance function
    # using numpy argsort to sort the distances and fetching their indeces
    indexes = np.argsort(dists)[:k] 
    # getting the nearest class lables 
    close_lables = [class_lables[i][0] for i in indexes]
    # finding the class lable to which the query instance belongs
    common_lable = Counter(close_lables).most_common(1)
    return common_lable[0][0]
    
        
# this function will calculate the accuracy of our distance weighted KNN classification.
def calculate_accuracy(true_class, pred_class):
    # comparing the predicted class lables with the true class lables and calculating the accuracy
    overall_accuracy = np.sum(true_class == pred_class)/len(true_class)
    # finding the unique classes in the true lable
    unique_classes = np.unique(true_class)
    ind_class_PA = {}
    for uc in unique_classes:
        # fetch the indexes where the true class belongs to a particular class
        indeces = np.where(true_class == uc)[0]
        # calculating the accuracy for each class
        ind_class_PA[uc] = np.sum(pred_class[indeces] == true_class[indeces])/len(indeces)
    
    return overall_accuracy, ind_class_PA

if __name__ == '__main__':
    
    # assigning the values to the hyper parameters
    a,k = 1,3
    
    # loading the training and test data from csv files as numpy arrays
    train_data = np.genfromtxt('trainingData.csv',delimiter=',')
    test_data = np.genfromtxt("testData.csv",delimiter=',')
    
    # seperating the feature data and class lables in training data using column indexing with numpy
    train_feature = train_data[:, 0:len(train_data[0])-1]
    train_class = train_data[:, len(train_data[0])-1:]
    
    
    # seperating the feature data and class lables in test data using column indexing with numpy
    test_query = test_data[:, 0:len(test_data[0])-1]
    test_class = test_data[:, len(test_data[0])-1:]
    test_class = test_class.reshape(len(test_class))
    
    # predicting the lables
    predict_lables = np.array([predict_class(train_feature,train_class,tq ,a,k) for tq in test_query])
    # calculating the accuracy
    total_acc, class_acc= calculate_accuracy(test_class, predict_lables)
    
    print("Over all Accuracy:", np.round(total_acc*100,2),end="%\n")
    print("Individual Class Accuracy:")
    for key,val in class_acc.items():
        print("Class -",key,":",np.round(val*100,2),end="%\n")
