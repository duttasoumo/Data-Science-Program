#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 17:08:15 2018

@author: gyan
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
'''from mpl_toolkits.mplot3d import Axes3D'''
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
fig= plt.figure()
'''a= fig.add_subplot(111, projection= '3d')'''
iris = datasets.load_iris()
x = pd.DataFrame(iris.data)
x.columns= ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
print(x)
a = KMeans(n_clusters=3)
a.fit(x)
a.labels_
#print(a)#
'''Z=[]'''
color = np.array(['Red', 'Black', 'Blue'])
z= plt.scatter(x.petal_length,x.sepal_length, c= color[a.labels_])
plt.title('Clustering Graph')
plt.rcParams['figure.figsize']=(16,14)
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
'''plt.Zlabel('Z-Axis')'''
plt.legend()
#print(z)#
n=accuracy_score(iris.target,a.labels_)
print(n)