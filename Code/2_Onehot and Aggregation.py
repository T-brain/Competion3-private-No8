import numpy as np
import pandas as pd

trainset = pd.read_csv('..../training-adj.csv')
testset = pd.read_csv('..../testing-adj.csv')

cl = pd.read_csv('..../claim_adj.csv')
pl = pd.read_csv('..../policy_adj.csv')


########################################################################

for i in pl.columns[pl.dtypes == object]:
    if sum( pl[i].value_counts()/pl[i].count() > 0.05 ) == 0 :
        pl.drop(i, axis = 1, inplace = True)
        print("feature '%s' has low frequecy in each levels.(under 5 percent)"%(i))


for i in cl.columns[cl.dtypes == object]:
    if sum( cl[i].value_counts()/cl[i].count() > 0.05 ) == 0 :
        cl.drop(i, axis = 1, inplace = True)
        print("feature '%s' has low frequecy in each levels.(under 5 percent)"%(i))
        
########################################################################
        
    
claim = pd.get_dummies(cl)

c1 = claim[claim.columns[0:9]].groupby('Policy_Number').mean()
c2 = claim[claim.columns[9:]].groupby(claim.Policy_Number).sum()
c1.columns = ['C_'] + c1.columns + ['_mean']
c2.columns = ['C_'] + c2.columns + ['_sum']


c1['C_Counts'] = np.array(claim['Policy_Number'].groupby(claim.Policy_Number).count())     
cc = pd.read_csv("C:/Users/a0972/Desktop/T-Brain/claim_0702.csv")

non_claim = (set(pl.Policy_Number)^set(cl.Policy_Number[0:69613]))&set(pl.Policy_Number)


c1.C_Counts[non_claim] = 0




########################################################################

policy = pd.get_dummies(pl)

p1 = policy[policy.columns[0:16]].groupby('Policy_Number').mean()
p2 = policy[policy.columns[16:300]].groupby(policy.Policy_Number).sum()
p3 = policy[policy.columns[300:500]].groupby(policy.Policy_Number).sum()
p4 = policy[policy.columns[500:700]].groupby(policy.Policy_Number).sum()
p5 = policy[policy.columns[700:900]].groupby(policy.Policy_Number).sum()
p6 = policy[policy.columns[900:]].groupby(policy.Policy_Number).sum()
p1.columns = ['P_'] + p1.columns + ['_mean']
p2.columns = ['P_'] + p2.columns + ['_sum']
p3.columns = ['P_'] + p3.columns + ['_sum']
p4.columns = ['P_'] + p4.columns + ['_sum']
p5.columns = ['P_'] + p5.columns + ['_sum']
p6.columns = ['P_'] + p6.columns + ['_sum']

p1['P_Counts'] = np.array(policy['Policy_Number'].groupby(policy.Policy_Number).count())   


########################################################################

cp = pd.concat([c1, p1, c2, p2, p3, p4, p5, p6], axis = 1)

train = cp.iloc[trainset.Policy_Number,:]
test = cp.iloc[testset.Policy_Number,:]

full = train.append(test, ignore_index = True)


full['Claim_per_Policy'] = full['C_Counts'] / full['P_Counts'] 
full['Total_Premium'] = full['P_Premium_mean'] * full['P_Counts'] 
full['Total_Deductible'] = full['C_Deductible_mean'] * full['C_Counts']
full['Quit'] = np.array([0] * full.shape[0])
full.Quit[full['P_Coverage_Deductible_if_applied_mean'] < 0] = 1

cols = full.columns.tolist()
cols = cols[-4:] + cols[:-4]
full = full[cols] 


trainset['Previous_Premium'] = np.array(full['Total_Premium'][0:210763])
testset['Previous_Premium'] = np.array(full['Total_Premium'][210763:])

trainset.to_csv('..../training-adj.csv', index = False)
testset.to_csv('..../testing-adj.csv', index = False)
full.to_csv('..../dataset1.csv', index = False)


