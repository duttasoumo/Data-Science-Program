import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class LogisticRegression1:
    def __init__(self, lr=0.01, num_iter=300000):
        self.lr = lr
        self.num_iter = num_iter
        
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __log(self, h):
#        print (h[0])
        for i in range(h.shape[0]):
            if not (h[i] == 0):
                 h[i] = np.log(h[i])
        return h
    
    def __loss(self, h, y):
        return ((-1/y.size)*(np.sum((y*self.__log(h)) + ((1-y)*(self.__log(1-h))))))
    
    def fit(self, X, y):
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        self.bias = 0
        costs = []
        x = []
        for i in range(self.num_iter):
            z = np.dot(X, self.theta) + self.bias
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            db = (np.sum(h-y)) / y.size
            self.theta -= self.lr * gradient
            self.bias -= self.lr * db
           # if(i%1 == 0):
            #    x.append(i)
             #   costs.append(self.__loss(h,y))
        return x, costs
    def predict_prob(self, X):
        return self.__sigmoid(np.dot(X, self.theta)+self.bias)
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold
    
    def score(self, Y_, y, sample_weight=None):
        return accuracy_score(y, Y_, sample_weight=sample_weight)
 
# Files
DATA_SET_PATH = "anes_dataset.csv"
 
def dataset_headers(dataset):
    return list(dataset.columns.values)
 
def train_logistic_regression(train_x, train_y):
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(train_x, train_y)
    return logistic_regression_model
 
def model_accuracy(trained_model, features, targets):
    accuracy_score = trained_model.score(features, targets)
    return accuracy_score
 	
def main():
    # Load the data set for training and testing the logistic regression classifier
    dataset = pd.read_csv(DATA_SET_PATH)
    print ("Number of Observations :: ", len(dataset))
 
    # Get the first observation
    print (dataset.head())
 
    header = dataset_headers(dataset)
    print ("Data set headers :: {headers}".format(headers=header))
 
    training_features = header[:-1]
    target = header[-1]
 
    # Train , Test data split
    train_x, test_x, train_y, test_y = train_test_split(dataset[training_features], dataset[target], train_size=0.7)
    print ("train_x size :: ", train_x.shape)
    print ("train_y size :: ", train_y.shape)
 
    print ("test_x size :: ", test_x.shape)
    print ("test_y size :: ", test_y.shape)
    
    # Training Logistic regression model
    trained_logistic_regression_model = train_logistic_regression(train_x, train_y)
    
    train_accuracy = model_accuracy(trained_logistic_regression_model, train_x, train_y)
 
    # Testing the logistic regression model
    test_accuracy = model_accuracy(trained_logistic_regression_model, test_x, test_y)
 
    print ("Train Accuracy(Library) :: ", train_accuracy)
    print ("Test Accuracy(Library) :: ", test_accuracy)
    
    model = LogisticRegression1()
    x, costs = model.fit(train_x, train_y)
    preds = model.predict(train_x,0.5)
   # plt.plot(x,costs)
    #plt.show()
# accuracy
    accuracy_score = model.score(preds, train_y)
    print ("Train Accuracy(User Function) :: ", accuracy_score)
    accuracy_score = model.score(model.predict(test_x,0.5), test_y)
    print ("Test Accuracy(User Function) :: ", accuracy_score)
if __name__ == "__main__":
    main()
