import pandas as pd
import numpy as np

#veri yukleme
veriler = pd.read_csv('veriler.csv')

#print(veriler)

ulke = veriler.iloc[:,0:1].values
#print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
ulke[:,0] =le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)