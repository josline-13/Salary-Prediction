import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 

data= pd.read_csv('Position_Salaries.csv')

x= data.iloc[:,1:2].values
y= data.iloc[:,2].values
plt.scatter(x,y,color='red')

#%matplotlib auto to get the graph out

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

linRegressor = LinearRegression()
linRegressor.fit(x,y)

#need to find the polynomial features of data 
#to visualize the data

plt.scatter(x,y,color='blue')
plt.plot(x,linRegressor.predict(x),color='red')
plt.show()
#this gives us a straight line so inorder to plot a line as a curve we change the degree of the variable
 
 
#instead of linear regressor we need polynomial regressor
#polynomialfeatures needs to specify degree by default it is 2
polyFeatures= PolynomialFeatures(degree=6)
#convert data to higher degree so that it becomes flexible to plot the line
newX = polyFeatures.fit_transform(x)
polyFeatures.fit(newX,y)

linRegressorNew = LinearRegression()
linRegressorNew.fit(newX,y)
 #convert x to second degree
#only need to convert independent variable interms of polynomail
plt.scatter(x,y,color='red')
plt.plot(x,linRegressorNew.predict(newX),'b')

y_pred=linRegressorNew.predict(newX)

plt.scatter(x,y,color='blue')
plt.plot(x,linRegressor.predict(x),color='red')
plt.show()

#to automate degree of polynomial stop when score is 0.99

score=0
l=[]
s=[]
i=2
while score <= 0.999:
        polyFeatures= PolynomialFeatures(degree=i)
        newX = polyFeatures.fit_transform(x)
        linRegressorNew = LinearRegression()
        linRegressorNew.fit(newX,y)
        l.append(linRegressorNew.predict(newX))
        s.append(linRegressorNew.score(newX,y))
        score=linRegressorNew.score(newX,y)
        i=i+1
        
q=1
for t in l:
    plt.subplot(1,i,q)
    plt.scatter(x,y,color='blue')
    plt.plot(x,l[q-1],'g--')
    
    
    plt.title("score={0},degree={1}".format(round(s[q-1],4),q+1))
    plt.show()
    q=q+1
        


#%matplotlib auto
    



sample = []
sample.append(2)




#to predict the salary of level 3.5
z=[3.5]
news= np.array(z).reshape(1,-1)
y_newpred=linRegressor.predict(news)
news = polyFeatures.fit_transform(news)







