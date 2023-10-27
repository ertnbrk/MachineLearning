'''
K-en yakın komşu 

K en yakın komşu işleyiş;
K-en yakın komşu Aranan y değerimizin x değerleri için yani aranan bağımlı değişkenin bağımsız değişkenlerinin diğer bağımsız değişkenlere olan uzaklığı bulunur
k sigma j=0 (xi - yi)^2 
Bu uzaklıklardan K tanesi seçilir(İsmi burdan gelmekte)
Seçilen K adet Y değerinin ortalaması alınarak sonuç bulunmuş olur 
:)

'''

#kütüphaneler
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
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state = 0)


#MODELİN OLUŞTURULMASI

#X_train'den y_train'i öğren 
knn_model = KNeighborsRegressor().fit(X_train,y_train)

print(dir(knn_model)) #knn_model içinden alabiliceklerimizi yazdır
#Tahminimizi yapıyoruz
y_predict = knn_model.predict(X_test)
#Hata kareler ortalmamızı yazdırıyoruz
print("\nHATA KARELER ORTALAMASI\n")
print(np.sqrt(mean_squared_error(y_test, y_predict)))

RMSE = []
# EL YORDAMIYLA K SAYISININ BELİRLENMESİ 
# Her k değeri için hata kareler ortalamasını alıoruz
for k in range(10):
    k = k+1
    knn_model = KNeighborsRegressor(n_neighbors=k).fit(X_train,y_train)
    y_pred = knn_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    RMSE.append(rmse)
    print("k = ",k, "için RMSE değeri : ",rmse)
    

#Otomatik yol ile K sayısının belirlenmesi 
'''GridSearchCV --> Hiperparemetrelerin değerlerini belirlememiz için kullanılan fonksiyondur'''

# 1'den 30 a kadar 1 er 1 er artarak n_neighborsa uzaklıkları hesapla ve k değerini crossvalidation ile bul
knn_params = {"n_neighbors":np.arange(1,30,1)}
knn = KNeighborsRegressor()
#ilk arguman modelimiz 2. arguman denenecek parametreler 3. arguman olarak 10 katlı çapraz doğrulama veriyoruz
knn_cv_model = GridSearchCV(knn, knn_params,cv=10).fit(X_train,y_train)
#K'NIN IDEAL DEGERİS
print("\nK için ideal değer = ",knn_cv_model.best_params_)

'''Bulduğumuz parametre değerini kullanarak final modelini oluşturuyoruz'''

#best param içnden seçme yapıyoruz çünkü birden fazla bağımsız değişken verseydik patlardık her ihtimale karşı
knn_tuned = KNeighborsRegressor(n_neighbors= knn_cv_model.best_params_['n_neighbors']).fit(X_train,y_train)
y_predict = knn_tuned.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))

print("Test Hatası karesi = >",rmse)


