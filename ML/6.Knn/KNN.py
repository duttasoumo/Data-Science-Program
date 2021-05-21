
# coding: utf-8

# # K- Nearest Neighbour by Smriti Anand and Abhirup Nandy

# Importing Libraries
# 

# In[6]:


import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd


# Importing Dataset into a pandas dataframe 

# In[8]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names) 

dataset.columns


# Preprocessing the data into training and testing data : 
# The next step is to split our dataset into its attributes and labels.

# In[9]:


X = dataset.iloc[:, :-1].values  
y = dataset.Class

print(y)
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# Feature Scaling -  Normalizing the features so t

# In[10]:


from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  


# Training and Prediction

# In[11]:


from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=5,metric='euclidean')  
classifier.fit(X_train, y_train)  


# In[12]:


y_pred = classifier.predict(X_test)  


# Evaluating the Algorithm
# 

# In[13]:


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))

