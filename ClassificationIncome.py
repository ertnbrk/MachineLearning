import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("Adult.csv")


missingValues = data.isnull().sum().sum()
print(missingValues)

'''Eksik verilerin giderilmesi'''

# Veriyi NumPy dizisine dönüştürün
veri_np = data.to_numpy()
# Veri dizisindeki "?" karakterlerini sayın
sayac = np.sum(veri_np == "?")
print(f"Veri setinizde {sayac} adet '?' karakteri bulunuyor.")

print(len(data.columns))

from sklearn.impute import SimpleImputer

'''Educational-num,occupation,workclass,gender,hours-per-week'''

for i in data.columns:
    print((data[i] == '?').sum())

    
print(data.shape)

data.replace('?',np.nan,inplace=True)

print(data.isnull().sum())

df = data.dropna()

X = df.drop('income',axis=1)
y = df['income']

country = X['native-country']
gender = X['gender']
occupation = X['occupation']
workclss = X['workclass']

'''Encode X'''

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

labelEncoder_X = LabelEncoder()
#LabelEncode
X['gender'] = labelEncoder_X.fit_transform(X['gender'])
X['race'] = labelEncoder_X.fit_transform(X['race'])
X['occupation'] = labelEncoder_X.fit_transform(X['occupation'])
X['workclass'] = labelEncoder_X.fit_transform(X['workclass'])
#Şimdilik kullanmayacağım özellikleri çıkarıyorum
X.drop(['marital-status','relationship','education','race','native-country'],axis=1,inplace=True)
#X.drop('workclass',axis=1,inplace=True)

'''Classifaction'''
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

randomModel = RandomForestClassifier(n_estimators=200,random_state=42)

randomModel.fit(X_train,y_train)

y_pred = randomModel.predict(X_test)


#Score
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Model Doğruluğu: {accuracy}")
print("Sınıflandırma Raporu:")
print(classification_rep)


