
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 01:12:37 2022

@author: diren
"""


#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('satislar.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)
#veri on isleme

aylar = veriler[['Aylar']]
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)

satislar2 = veriler.iloc[:,:1].values
print(satislar2)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

#dogrusal regresyon

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()



plt.plot(x_train,y_train)
plt.plot(x_test,tahmin)






