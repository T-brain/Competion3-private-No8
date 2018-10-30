import numpy as np
import pandas as pd

from sklearn import preprocessing
le = preprocessing.LabelEncoder()


trainset = pd.read_csv("..../training-adj.csv")
testset = pd.read_csv("..../testing-adj.csv")
pl = pd.read_csv("..../policy_adj.csv")

for i in pl.columns[pl.dtypes == object] :
   kk = pl[i].value_counts()/pl[i].count()*100
   if sum(kk >= 5) == 0 :
       pl = pl.drop(labels = i, axis = 1)
       print("feature '%s' has low frequecy in each levels.(under 5 percent)"%(i))


for i in pl.columns[pl.dtypes == object] :
    pl[i] = pl[i].fillna('None')
    

cat = pl.columns[pl.dtypes == object]

for i in cat :
    feature = i + '_mean_encode_con'
    mean_encode = pl.Premium.groupby(pl[i]).mean()
    kkk = dict(zip(mean_encode.index,list(mean_encode)))
    pl[feature] = pl[i]
    for j in mean_encode.index :            
        pl[feature].replace(j, kkk[j], inplace = True)
       
pp = pl[pl.columns[-19:]]

ppp = pp.groupby(pl.Policy_Number).mean()
    
tr = ppp.iloc[trainset.Policy_Number,:]
te = ppp.iloc[testset.Policy_Number,:]

full3 = tr.append(te, ignore_index = True)

full3.to_csv('..../dataset3.csv', index = False)

