import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from model import *
import os
from typing import List, Callable, Tuple
from config import *
import k_means_tsp
import sklearn

def read_data(file_location: str):
    if os.path.isfile(file_location):
        with open(file_location) as file: # Use file to refer to the file object
            test_case_count = int(file.readline())
            test_case = list()
            for _ in range(test_case_count):
                city_count = int(file.readline())
                cities = list()
                for count in range(city_count):
                    #line = file.readline().split(',')
                    #x, y = list(map(lambda x: float(x), line))
                    x,y=file.readline().strip().split()[1:]
                    cities.append([float(x),float(y)])
                optimal = float(file.readline())
                
            
        return cities

def divide_list(data,clas,num_class):
    sub_class=[]
    subs_class=[]
    for i in range(num_class):
        for j in range(len(data)):
                if i==clas[j]:
                       sub_class.append(data[j])
        subs_class.append(sub_class)
        sub_class=[]       
    return subs_class   


def main_kmeans(X,num):
     np.random.seed(0)
     batch_size = 45
     #X=read_data(DATA_FILE)
     #X=np.array(X)
     #centers = [[1, 1], [-1, -1], [1, -1],[1, 0.5], [0.5, -1], [0.5, 1],[0.5, 0.5]]
     #centers = [[1, 1], [-1, -1], [1, -1],[-1, 1]]
     centers = [[1, 1], [-1, -1], [1, -1]]
     #centers = [[1, 1], [-1, -1]]
     n_clusters = num
     #X, labels_true = make_blobs(n_samples=52, centers=centers, cluster_std=0.7)
     #X=read_data(DATA_FILE)
     #X=np.array(X)
     #labels_true=

     k_means = KMeans( init='k-means++',n_clusters=num, n_init=10)
     k_means.fit(X)
     k_means_labels = k_means.labels_
     #k_means_labels= np.array([1, 1, 1, 0, 0, 0, 1, 3, 3, 3, 2, 2, 2, 2, 0, 1, 1, 1, 3, 1, 1, 1, 1, 0, 0, 2, 2, 2, 1, 1, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 1, 3, 0, 1, 0, 2, 0, 0, 1, 2, 2])
     k_means_cluster_centers = k_means.cluster_centers_
     k_means_labels_unique = np.unique(k_means_labels)
     #print("aymen",k_means_cluster_centers,)
     
     #colors = ['#4EACC5', '#FF9C34', '#4E9A06','red','green','blue','yellow']ter
     colors = ['#4EACC5', '#FF9C34', '#4E9A06',]
     s=divide_list(X,k_means_labels,3)
     #print(s)
     #plt.figure()
     clusters=[]
     id=[]
     #plt.hold(True)   
     for k, col in zip(range(n_clusters), colors):
         my_members = k_means_labels == k
         cluster_center = k_means_cluster_centers[k]
         clusters.append(X[my_members])
         id.append(my_members)
         '''
         plt.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='o')
         plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
         
     
     plt.title('KMeans')    
     plt.grid(True)
     plt.show()
     '''

     return clusters
      
    
##############################################################################
# Generate sample data
''''
np.random.seed(0)
batch_size = 45
X=read_data(DATA_FILE)
X=np.array(X)
print(type(X))
#centers = [[1, 1], [-1, -1], [1, -1],[1, 0.5], [0.5, -1], [0.5, 1],[0.5, 0.5]]
centers = [[1, 1], [-1, -1], [1, -1],[-1, 1]]
n_clusters = len(centers)
#X, labels_true = make_blobs(n_samples=52, centers=centers, cluster_std=0.7)
X=read_data(DATA_FILE)
X=np.array(X)
#labels_true=
print(type(X))

##############################################################################
# Compute clustering with Means

k_means = KMeans(init='k-means++', n_clusters=4, n_init=10)
k_means.fit(X)
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels_unique = np.unique(k_means_labels)



#colors = ['#4EACC5', '#FF9C34', '#4E9A06','red','green','blue','yellow']
colors = ['#4EACC5', '#FF9C34', '#4E9A06','red']
s=divide_list(X,k_means_labels,4)
print(s)
plt.figure()
#plt.hold(True)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
plt.title('KMeans')    
plt.grid(True)
plt.show()
'''

'''
cx=read_data(DATA_FILE)
cxx=np.array(cx)
v=main_kmeans(cxx)
print(v)

'''