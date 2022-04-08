# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:37:43 2022

@author: diren
"""


import pandas as pd
import numpy as np

#veri yukleme
veriler = pd.read_csv('veriler.csv')

print(veriler)

#encoder: Kategorik -> Numeric

ulke = veriler.iloc[:,0:1].values
print(ulke)
Yas = veriler.iloc[:,1:4].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
ulke[:,0] =le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#encoder: Kategorik -> Numeric

c = veriler.iloc[:,-1:].values
print(c)

c[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(c)

c = ohe.fit_transform(c).toarray()
print(c)


# numpy dizileri dataframe dönüşümü

sonuc = pd.DataFrame(data=ulke, index= range(22), columns=['fr','tr','us'])

sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns=['boy','kilo','yas'])

cinsiyet = veriler. iloc[:,-1].values #son kolonu aldık.

sonuc3= pd.DataFrame(data= c[:,:1], index = range(22), columns=['cinsiyet'])

#dataframe birleştirme işlemi
s  = pd.concat([sonuc, sonuc2], axis = 1)

print(s)
s2=pd.concat([s,sonuc3],axis = 1)
print(s2)

#verilerin eğitim test için bölünmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(s,sonuc3,test_size = 0.33,random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#Boy kolonu tahmini
boy = s2.iloc[:,3:4].values
print(boy)

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis = 1)

x_train, x_test, y_train,y_test = train_test_split(veri,boy,test_size = 0.33,random_state = 0)

regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)


#backward elemination 
import statsmodels.api as sm

X = np.append(arr = np.ones((22,1)).astype(int),values = veri,axis = 1)

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype= float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())
