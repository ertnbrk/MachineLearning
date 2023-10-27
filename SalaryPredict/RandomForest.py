"""RANDOM FOREST"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Veri Yukleme
veriler = pd.read_csv('maaslar.csv')

#veriyi bölüyorum 
x = veriler.iloc[:,1:2]  #eğitim seviyesi
y = veriler.iloc[:,2:]  #maaş

X = x.values
Y = y.values

from sklearn.ensemble import RandomForestRegressor
# n_estimators kaç tane decission tree çiziliceğini veriyoruzs
rf_reg = RandomForestRegressor(random_state=0,n_estimators=10)

rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.5]]))

plt.scatter(X, Y, color='red')
plt.plot(X,rf_reg.predict(X),color='blue')

Z = X + 0.5

plt.plot(X,rf_reg.predict(Z),color='green')


'''
R^2 ile score hesaplama yapıcaz yani tahmin ve gerçek değerleri arasındaki bağlantıyı bulur r^2 0 veya negatif ise sıçtın
'''

from sklearn.metrics import r2_score

print(r2_score(Y, rf_reg.predict(X)))