'''Decission Tree Regression'''




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


'''Data Processing'''
veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]  #eğitim seviyesi
y = veriler.iloc[:,2:]  #maaş

X = x.values
Y = y.values

'''Linear Model'''
from sklearn.linear_model import LinearRegression

linearModel = LinearRegression()

linearModel.fit(X,Y)

linear_predict = linearModel.predict(X)

'''Polynomial Model'''


from sklearn.preprocessing import PolynomialFeatures

PolyModel = PolynomialFeatures(degree=4) 

x_poly = PolyModel.fit_transform(X)

print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

'''Predict & Visualization'''

plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(PolyModel.fit_transform(X)),color='blue')
plt.show()
#Scaling
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
sc2 = StandardScaler()

x_olcekli = sc1.fit_transform(X)
y_olcekli = sc1.fit_transform(Y)

#SVM

from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')

svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli, svr_reg.predict(x_olcekli),color='blue')
plt.show()
print(svr_reg.predict([[11]]))


from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0) 

r_dt.fit(X,Y)
plt.scatter(X,Y ,color="green")
plt.plot(X,r_dt.predict(X),color="purple")

