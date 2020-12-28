  
#kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#kod bolumu

#veri yukleme


veriler = pd.read_csv('eksikveriler.csv')

#sci - kit learn  ekik veriler

from sklearn.impute import SimpleImputer
 
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

Yas = veriler.iloc[:,-2:-1].values


imputer = imputer.fit(Yas)    #fit eğitmek için kullanılır  kolonların ortalama değerlerini öğrenecek

Yas = imputer.transform(Yas)          #transformla öğrendiğini yerine koy
print(Yas)

veriler.iloc[:,-2:-1] = Yas  #değiştirilmiş veriler ile yer değiştir. 

print(veriler)
