#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:44:32 2018

@author: neetu
"""

from sklearn import datasets

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

dataset = datasets.load_iris()
model = GaussianNB()
model.fit(dataset.data, dataset.target)
print(model)
expected = dataset.target
predicted = model.predict(dataset.data)

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
fig = plt.figure()
p = pd.DataFrame(dataset.data)
p.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
print(p)
color = np.array(['Red', 'Black', 'Blue'])
z = plt.scatter(p.sepal_length, p.petal_length, c = color[predicted] )
plt.title('Clustering Graph')
plt.rcParams['figure.figsize']= (12,12)
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
from sklearn.metrics import accuracy_score
n = accuracy_score(expected,predicted)
print(n)


