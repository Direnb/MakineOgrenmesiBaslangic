import pandas as pd
import numpy as np

#veri yukleme
veriler = pd.read_csv('veriler.csv')

print(veriler)

ulke = veriler.iloc[:,0:1].values
#print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
ulke[:,0] =le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

Yas = veriler.iloc[:,1:4].values

# concat

sonuc = pd.DataFrame(data=ulke, index= range(22), columns=['fr','tr','us'])

sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns=['boy','kilo','yas'])

cinsiyet = veriler. iloc[:,-1].values #son kolonu aldık.

sonuc3= pd.DataFrame(data= cinsiyet, index = range(22), columns=['cinsiyet'])

s  = pd.concat([sonuc, sonuc2], axis = 1)

print(s)
