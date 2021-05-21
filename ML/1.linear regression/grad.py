

import pandas as pd
def gradient_descent(x,y):
    m = b = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.01
    
    for i in range(iterations):
        y_predicted = m * x + b
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m = m - learning_rate * md
        b = b - learning_rate * bd
        print ("m={} , b={} , cost={} iteration={}".format(m,b,cost, i))
# Reading Data
data = pd.read_csv('Salary_Data.csv')
#print(data.shape)
#data.head()



# Collecting X and Y
x = data['YearsExperience'].values
y = data['Salary'].values



gradient_descent(x, y)