import numpy as np
import pandas as pd
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

trainset = pd.read_csv("..../training-adj.csv")
testset = pd.read_csv("..../testing-adj.csv")

pl = pd.read_csv("..../policy_adj.csv")


le.fit(pl['Vehicle_Make_and_Model1'])
pl['Vehicle_Make_and_Model1'] = le.transform(pl['Vehicle_Make_and_Model1'] )

le.fit(pl['Distribution_Channel'])
pl['Distribution_Channel'] = le.transform(pl['Distribution_Channel'] )

le.fit(pl['iply_area'])
pl['iply_area'] = le.transform(pl['iply_area'] )

pl['Vehicle_Make_and_Model1'] = pl['Vehicle_Make_and_Model1'].apply(str)
pl['Distribution_Channel'] = pl['Distribution_Channel'].apply(str)
pl['iply_area'] = pl['iply_area'].apply(str)

for i in pl.columns[pl.dtypes == object]:
    if sum( pl[i].value_counts()/pl[i].count() > 0.05 ) == 0 :
        pl.drop(i, axis = 1, inplace = True)
        print(' %s has been removed'%(i))
        
for i in pl.columns[pl.dtypes == object] :
    pl[i] = pl[i].fillna('None')
    
cat_pl = pl.columns[pl.dtypes == object]



cat_pl = ['Cancellation', 'Vehicle_Make_and_Model1', 'Imported_or_Domestic_Car',
       'Main_Insurance_Coverage_Group', 'Insurance_Coverage',
       'Distribution_Channel', 'fassured', 'fsex', 'fmarriage', 'iply_area',
       'nequipment9']

policy = pd.merge(trainset, pl, on = 'Policy_Number')

policy['renewal'] = policy.Next_Premium
policy.renewal[policy.renewal > 0] = 1

    
a = policy[policy.renewal > 0]
b = policy[policy.renewal == 0]


pl = pl[cat_pl]

for i in range(0,len(cat_pl)):
    for j in range((i + 1),len(cat_pl)):
        k1 = (a[cat_pl[i]] + a[cat_pl[j]]).value_counts()/a[cat_pl[i]].count() * 100
        k2 = (b[cat_pl[i]] + b[cat_pl[j]]).value_counts()/b[cat_pl[i]].count() * 100
        if (abs(k1 - k2).sum())/len(k1) >= 0.5:
            k = '%s_%s'%(cat_pl[i],cat_pl[j])
            pl[k] = pl[cat_pl[i]] + pl[cat_pl[j]]
            print(k)

pl = pl[pl.columns[11:]]

pp = pd.get_dummies(pl)

policy = pd.read_csv("..../policy_adj.csv")

p1 = pp.groupby(policy.Policy_Number).sum()


tr = p1.iloc[trainset.Policy_Number,:]
te = p1.iloc[testset.Policy_Number,:]

cat_pl_sum = tr.append(te, ignore_index = True)

cat_pl_sum.to_csv('..../dataset2.csv', index = False)
