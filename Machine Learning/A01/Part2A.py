# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 18:33:15 2021

@author: reddy
"""
# importing the numpy and matplotlib 
import numpy as np
import matplotlib.pyplot as plt
from Part1A import *

# this function will inititalize the required number of centroids
def initialize_centroids(feature_data, k):
    # select k number of random centroids from feature data
    ind = np.random.choice(feature_data.shape[0], size=k)
    rand_cent = feature_data[ind]
    return rand_cent

# this function will iterate through the feature data and find the closest centroids to the instances.
def assign_centroids(feature_data, centroids):
    count = 0
    dists = []
    centroid_indeces = []
    while len(feature_data) > count: # iterating through the feature data
        # calculating the distances between the feature data and the centroids
        dists = np.argmin(minkowski_distance(centroids,feature_data[count],1))
        # collecting all the cluster indeces to which the feature data belongs 
        centroid_indeces.append(dists)
        count +=1
        
    return centroid_indeces

# This function will move the centroids to their new positions
def move_centroids(feature_data, centroid_indeces,curr_centroids):
    count = 0
    new_centroids = []        
    while len(curr_centroids) > count: # iterating through the centroids
        # calculating the mean value of the feature data to assign new position for centroids
        cent = np.mean(feature_data[np.array(centroid_indeces) == count],axis=0)   
        # collecting the new centroids
        new_centroids.append(cent)
        count += 1
    return new_centroids
# This function will calculate the current distortion cost
def calculate_cost(feature_data, centroid_indeces,curr_centroids):
    # calculating the distortion cost based on the given formula in the assignment
    cost =(np.sum(np.square(np.sum(np.square(curr_centroids[centroid_indeces] - feature_data),axis =1))))
    return cost

# this method will run for the specified number of restarts and returns the best centroids and its cost
def restart_KMeans(feature_data, k, iterations, restarts):
    best_cost = []
    best_centroids = []
    for i in range(restarts): # running through the given restarts
        curr_centroids = np.array(initialize_centroids(feature_data,k))
        for j in range(iterations): # running through given iterations
            centroid_indeces = np.array(assign_centroids(feature_data, curr_centroids))
            curr_centroids =  np.array(move_centroids(feature_data,centroid_indeces,curr_centroids))
        cost = calculate_cost(feature_data,centroid_indeces,curr_centroids)
        # collecting the best cost and centroids from each restart
        best_cost.append(cost)
        best_centroids.append(curr_centroids)
        
    return best_centroids[np.argsort(best_cost)[0]], best_cost[np.argsort(best_cost)[0]]

if __name__ == '__main__':
    
    # loading the data from csv file into a numpy array.
    feature_data = np.genfromtxt('clusteringData.csv',delimiter=',')
    
    k = 10
    iterations = 10
    restarts = 10
    inertia = []
    K = range(2,k)
    for i in K:
        best_centroids, best_cost = restart_KMeans(feature_data, i,iterations,restarts)
        inertia.append(best_cost)
        print(best_centroids,"\n",best_cost)
    
    plt.figure(figsize=(16,8))
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('K')
    plt.ylabel('Cost')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
