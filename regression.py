#importing required libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston

#dataset
boston=load_boston()
print(boston.DESCR)

#reading dataset
dataset=boston.data
for name,index in enumerate(boston.feature_names):
    print(index,name)
    
#reshaping
data=dataset[:,12].reshape(-1,1)
np.shape(dataset)
#target values
target=boston.target.reshape(-1,1)
np.shape(target)

#working of matplotlib
%matplotlib inline
plt.scatter(data,target,color='blue')
plt.xlabel('Lower income population')
plt.ylabel('Cost of house')
plt.show()

#regression
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(data,target)

#prediction
pred=reg.predict(data)

#working of prediction
%matplotlib inline
plt.scatter(data,target,color='red')
plt.plot(data,pred,color='green')
plt.xlabel('Lower income population')
plt.ylabel('Cost of house')
plt.show()

#circumventing curve issue
from sklearn.preprocessing import PolynomialFeatures
#pipeline
from sklearn.pipeline import make_pipeline

model=make_pipeline(PolynomialFeatures(3),reg)
model.fit(data,target)

pred=model.predict(data)

%matplotlib inline
plt.scatter(data,target,color='blue')
plt.plot(data,pred,color='red')
plt.xlabel('Lower income population')
plt.ylabel('Cost of house')
plt.show()

#R_2 metric
from sklearn.metrics import r2_score

r2_score(pred,target)

