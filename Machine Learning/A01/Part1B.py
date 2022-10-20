# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 05:36:22 2021

@author: reddy
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import SelectPercentile, f_classif
# importing the functions from Part1A python file
from Part1A import *


class k_NN_Algorithm:
    def __init__(self,choice):
        self.a = 1
        self.test_data = np.genfromtxt("testData.csv",delimiter=",")
        self.train_data = np.genfromtxt("trainingData.csv",delimiter=",")
        self.test_query = self.test_data[:, 0:len(self.test_data[0])-1]
        self.train_feature = self.train_data[:, 0:len(self.train_data[0])-1]       
        self.k = 3
        self.train_class = self.train_data[:, len(self.train_data[0])-1:]
        self.test_class = self.test_data[:, len(self.test_data[0])-1:]
        self.test_class = self.test_class.reshape(len(self.test_class))
        self.choice = choice
        if self.choice == "1":
            print()
            print("Standard KNN ")
              
        if self.choice == "2": 
        #SelectionPercentile
            print("Implementing Selection Percentile ")
            x = SelectPercentile(f_classif, percentile=20)
            print("Shape of training data before implementing selection percentile",self.train_feature.shape)
            self.train_feature = x.fit_transform(self.train_feature , self.train_class.reshape(len(self.train_class)))
            self.test_query = x.transform(self.test_query)
            print("Shape of training data after implementing selection percentile",self.train_feature.shape)
        elif self.choice == "3":
        # Normalization 
            print()
            print("Normalization ")
            self.test_query = preprocessing.normalize(self.test_query, norm='l2')
            self.train_feature = preprocessing.normalize(self.train_feature, norm='l2')
        elif self.choice == "4":
        #Scale Features 
            #MaxAbsScaler
            print()
            print("MaxAbsScaler ")
            maxabs = preprocessing.MaxAbsScaler()
            self.test_query = maxabs.fit_transform(self.test_query)
            self.train_feature = maxabs.fit_transform(self.train_feature)
        elif self.choice == "5":
            #MinMaxScaler
            print()
            print("MinMaxScaler ")
            minmax = preprocessing.MinMaxScaler()
            self.test_query = minmax.fit_transform(self.test_query)
            self.train_feature = minmax.fit_transform(self.train_feature)
        elif self.choice == "6":
            #StandardScaler
            print()
            print("StandardScaler ")
            scalerTD = preprocessing.StandardScaler().fit(self.test_query)
            scalerTD.transform(self.test_query)
            scalerTF = preprocessing.StandardScaler().fit(self.train_feature)
            scalerTF.transform(self.train_feature)

    def main(self):
        accuracy = []
        k = 10
        for i in range(3,k):
            predict_lables = np.array([predict_class(self.train_feature,self.train_class,tq ,self.a,i) for tq in self.test_query])
            total_acc, class_acc= calculate_accuracy(self.test_class, predict_lables)
            accuracy.append(total_acc)
        print("Accuracy for k-NN:\n",accuracy)
        plt.plot([k for k in range(3,10)] ,accuracy ,marker = 'o')
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        plt.title('K - NN Model')
        plt.show()


if __name__ == '__main__':
    
    choice = input("Enter the choice from bellow\n Standard KNN -> 1 \n SelectionPercentile -> 2\nNormalization -> 3 \n MaxAbsScaler -> 4 \n MinMaxScaler -> 5 \n StandardScaler -> 6\n")
    my_knn = k_NN_Algorithm(choice)
    my_knn.main()