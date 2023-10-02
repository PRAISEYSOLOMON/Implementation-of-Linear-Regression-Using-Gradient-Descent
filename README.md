# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1.Import the required library and read the dataframe.

 2.Write a function computeCost to generate the cost function.

 3.Perform iterations og gradient steps with learning rate.

 4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: PRAISEY S
RegisterNumber: 212222040117
# Import required package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
data
data.shape
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Vs Prediction")
def computeCost(X,y,theta):
    m=len(y)
    h=X.dot(theta)
    square_err=(h-y)**2

    return 1/(2*m) * np.sum(square_err)
data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)
theta.shape
y.shape
X.shape
def gradientDescent(X,y,theta,alpha,num_iters):
  
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions - y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta, J_history
  
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")
plt.plot(J_history)
plt.xlabel("Iternations")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Polpulation of City (10,000s)")
plt.ylabel("Profit (10,000s)")
plt.title("Profit Prediction")
def predict(x,theta):
  predictions= np.dot(theta.transpose(),x)
  return predictions[0]
  
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0))) 
*/
```

## Output:

READ CSV FILE:

![out 3 1](https://github.com/PRAISEYSOLOMON/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119394259/ea550585-d65d-4215-92a0-42c80d68e097)

DATASET SHAPE:

![out 3 2](https://github.com/PRAISEYSOLOMON/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119394259/9da7999a-ed12-4b05-957c-641c2484bf4f)

PROFIT VS PREDICTION:

![out 3 3](https://github.com/PRAISEYSOLOMON/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119394259/2c9d9c0a-9c30-4269-b6fd-671e294ccfef)

X , Y, THETA VALUE:

![out 3 4](https://github.com/PRAISEYSOLOMON/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119394259/8ebaa8cf-e191-4116-b8f5-1ad0aad28adb)

GRADIENT DESCENT :

![out 3 5](https://github.com/PRAISEYSOLOMON/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119394259/4d2a7fd6-bb32-4a99-a38d-bb7f0d72503c)

COST FUNCTION USING GRADIENT DESCENT GRAPH:

![out 3 6](https://github.com/PRAISEYSOLOMON/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119394259/9ede3154-3d1c-4488-b873-6b8fbf74e2d5)

PROFIT PREDICTION GRAPH:

![out 3 7](https://github.com/PRAISEYSOLOMON/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119394259/57d77fbb-927b-4925-aa3c-76807cf564cc)

PROFIT PREDICTION:

![out 3 8](https://github.com/PRAISEYSOLOMON/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119394259/3f89d0dc-b0cc-4fb0-b052-426ea22451a1)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
