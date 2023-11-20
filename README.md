# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program
2. import numpy as np. Give the header to the data.
3. Find the profit of population. Plot the required graph for both for Gradient Descent Graph and Prediction Graph.
4. End the program.

## Program:
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("/content/ex1.txt",header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Predication")
def computeCost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(x,y,theta)
def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=x.dot(theta)
    error=np.dot(x.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(x,y,theta))
  return theta,J_history
theta,J_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color='r')
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Prediction")
def predict(x,theta):
  predictions= np.dot(theta.transpose(),x)
  return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*1000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))
predict2=predict(np.array([1,7]),theta)*1000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```


## Output:
## Compute Cost Value:
![image](https://github.com/sarveshjustin/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497481/095b352d-381d-4a4d-bbaf-3e022f5bb515)
## h(x) Value:
![image](https://github.com/sarveshjustin/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497481/16ce37bb-d382-413d-a885-89fdc7c8552a)
## profit prediction graph
![image](https://github.com/sarveshjustin/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497481/a8c57dea-c370-405f-b5d2-201e632a13c5)
## cost function using gradient descent
![image](https://github.com/sarveshjustin/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497481/57a5d02b-3be1-4d3c-994d-7a86af7756d1)
## profit prediction graph
![image](https://github.com/sarveshjustin/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497481/3d75c8be-1c10-4a52-bd9c-653c89b73450)
## profit for population 35000
![image](https://github.com/sarveshjustin/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497481/71bb0e67-0af3-4f24-83e3-a2a0329c1b3f)
## profit for population 70000
![image](https://github.com/sarveshjustin/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497481/3dc1e38f-a896-477e-8ae4-cba07f78903f)
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
