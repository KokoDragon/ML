#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\progr\Downloads\data\dataset.csv")
plt.scatter(df.values[:,1], df.values[:,2], c = df['3'], alpha=0.8)


# In[64]:


df = df.values
np.random.seed(5)
np.random.shuffle(df)

train = df [0:int(0.7*len(df))]
test = df[int(0.7*len(df)): int(len(df))]

x_train = train[:,0:3]
y_train = train[:, 3]

x_test = test[:,0:3]
y_test = test[:,3]


# In[65]:


def perceptron_train (x,y,z,eta,t):
    w = np.zeros(len(x[0]))
    n = 0
                 
    yhat_vec = np.ones(len(y))
    errors = np.ones(len(y))
    J = []
    
    while n < t:
        for i in range(0, len(x)):
            f = np.dot(x[i], w)
            if f >= z:
                yhat = 1
            else:
                yhat = 0
            yhat_vec[i] = yhat
            
            for j in range(0, len(w)):
                w[j] = w[j] + eta*(y[i]-yhat)*x[i][j]
        
        n+=1
        for i in range(0, len(y)):
            errors[i] = (y[i] - yhat_vec[i]) ** 2
        J.append(0.5*np.sum(errors))
        
    
    return w, J
    
    
z = 0.0
eta = 0.1
t = 50
perceptron_train(x_train, y_train, z, eta, t)


# In[66]:


from sklearn.metrics import accuracy_score

w = perceptron_train(x_train, y_train, z, eta, t)[0]

def perceptron_test (x,w,z,eta,t):
    y_pred = []
    for i in range(0, len(x-1)):
        f = np.dot(x[i], w)
        if f > z:
            yhat = 1
        else:
            yhat = 0
        y_pred.append(yhat)
    return y_pred

y_pred = perceptron_test (x_test, w, z, eta, t)

print(y_pred)
print(y_test)


# In[ ]:




