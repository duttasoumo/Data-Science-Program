
# coding: utf-8

# # Scikit-learn Implementation

# ## Reading the dataset

# In[1]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

data_print = np.c_[X,y]
print('DataSet:\n', data_print)
print('Label Names:\n', list(enumerate(iris.target_names)))


# ### Visualizing the Dataset

# In[2]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
pos_0 = np.where(y == 0)[0]
pos_1 = np.where(y == 1)[0]
pos_2 = np.where(y == 2)[0]

plt.figure(figsize=(10, 10))
plt.plot(X[pos_0, 0], X[pos_0, 1], 'ro', label='setosa')
plt.plot(X[pos_1, 0], X[pos_1, 1], 'bo', label='versicolor')
plt.plot(X[pos_2, 0], X[pos_2, 1], 'go', label='virginica')

plt.title('Scatterplot of Iris-2D')
plt.xlabel('petal width')
plt.ylabel('petal length')

plt.legend(loc=2, prop={'size': 15})
plt.show()


# ## Splitting into Train and Test Sets

# In[3]:


(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = 0.3) #Test_Set = 25% by default

train_print = np.c_[X_train, y_train]
test_print = np.c_[X_test, y_test]

# print("Training Set:\n", train_print)
# print("Test Set:\n", test_print)


# ## Training The Decision Tree Classifier

# In[4]:


tree_clf = DecisionTreeClassifier(max_depth=3)
tree_clf.fit(X_train, y_train)


# ## Making Predictions on the Test Set

# In[5]:


# print(tree_clf.predict_proba(X_test))
y_predicted = tree_clf.predict(X_test)

print('Predicted Results:\n', y_predicted)
print('Actual Results:\n', y_test)



# ## Analysis

# In[6]:


from sklearn.metrics import classification_report, confusion_matrix  

total = len(y_test)
correct = len(np.where(y_predicted == y_test)[0])
print('Total Test Examples: ', total)
print('Correct: ', correct)

print('Accuracy:', correct/total * 100)

print('\n\nConfusion Matrix:\n', confusion_matrix(y_test, y_predicted))  
print('\n\nClassification Report:\n', classification_report(y_test, y_predicted))

