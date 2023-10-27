"""
XGboost GBM'in hız ve tahmin performansını arttırmak üzere optimize edilmiş ölçeklenebilir ve farklı platformlara entegre edilebilir halidir
"""

from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import scale , StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer


from warnings import filterwarnings
filterwarnings('ignore')
#veri yükleme

veri = pd.read_csv("Hitters.csv")
#eksik verileri ortadan kaldırdım
veri = veri.dropna()
#Kategorik verilerin dummy variable dönüştürmesi
dms = pd.get_dummies(veri[['League','Division','NewLeague']])
#Bağımlı değişkenin veri kümesine aktarılması
y = veri["Salary"]
#Bağımsız değişkenlerin testpiti ve veri kümesine aktarılması
X_ = veri.drop(["Salary","League","Division","NewLeague"],axis=1)
#Bağımsız değişken ile Bağımlı değişkenin dummy halinin veri kümesine aktarılması
X = pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
#train test e bölünme
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state = 49)

# Model ve Tahmin
xgb = XGBRegressor().fit(X_train,y_train)

y_pred = xgb.predict(X_test)

print(np.sqrt(mean_squared_error(y_test, y_pred)))

#Model Tuning

xgb = XGBRegressor()

xgb_params = {"learning_rate":[0.1,0.5],"max_depth":[2,3],"n_estimators":[100,200],"colsample_bytree":[0.4,0.7,1]}

xgb_cv_model = GridSearchCV(xgb, xgb_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)

best_params = xgb_cv_model.best_params_
xgb_tuned = XGBRegressor(learning_rate=best_params["learning_rate"],max_depth = best_params["max_depth"],n_estimators = best_params["n_estimators"],colsample_bytree=best_params["colsample_bytree"]).fit(X_train,y_train)

y_predict = xgb_tuned.predict(X_test)
print("RMSE ilkel olmayan\n",np.sqrt(mean_squared_error(y_test, y_predict)))

