import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

train = pd.read_csv("..../training-set.csv")
test = pd.read_csv("..../testing-set.csv")
claim = pd.read_csv("..../claim_0702.csv")
policy = pd.read_csv("..../policy_0702.csv")

FPN = train.Policy_Number.append(test.Policy_Number, ignore_index=True)

ALL = FPN.append(claim.Policy_Number, ignore_index = True)

ALL1 = ALL.append(policy.Policy_Number, ignore_index = True)

# label Policy_Number #

le = LabelEncoder()
kk = le.fit_transform(ALL1)

train.Policy_Number = kk[0 : 210763]
test.Policy_Number = kk[210763 : 351273]
claim.Policy_Number = kk[351273 : 420886]
policy.Policy_Number = kk[420886:]

# datetime transform to age #

def age1(born,recent) :
    if pd.isnull(born) :
        res = np.nan
    else :       
        b = datetime.strptime(born, '%m/%Y')
        r = datetime.strptime(recent, '%m/%Y')
        if r.month > b.month :
            res = ( r.year - b.year ) + round((r.month - b.month)/12,1)
        else :
            res = ( r.year - b.year - 1 ) + round((12 - b.month + r.month)/12,1)      
    return res

def age2(born,recent) :
    if pd.isnull(born) :
        res = np.nan
    else :       
        b = datetime.strptime(born, '%Y/%m')
        r = datetime.strptime(recent, '%m/%Y')
        if r.month > b.month :
            res = ( r.year - b.year ) + round((r.month - b.month)/12,1)
        else :
            res = ( r.year - b.year - 1 ) + round((12 - b.month + r.month)/12,1)      
    return res

def age3(born,recent) :
    if pd.isnull(born) :
        res = np.nan
    else :       
        res = recent - born
    return res

claim.DOB_of_Driver = [age1(x ,'07/2018') for x in claim.DOB_of_Driver]
claim.Accident_Date = [age2(x ,'07/2018') for x in claim.Accident_Date]
policy.Manafactured_Year_and_Month = [age3(x ,2018) for x in policy.Manafactured_Year_and_Month]
policy.ibirth = [age1(x ,'07/2018') for x in policy.ibirth]
policy.dbirth = [age1(x ,'07/2018') for x in policy.dbirth]

#################################################################

claim = claim.drop(labels = 'Claim_Number', axis = 1)

claim.Nature_of_the_claim.replace(1,'Pay', inplace = True)
claim.Nature_of_the_claim.replace(2,'Recovery', inplace = True)
pd.unique(claim.Nature_of_the_claim)

claim["Driver's_Gender"].replace(1,'M', inplace = True)
claim["Driver's_Gender"].replace(2,'F', inplace = True)
claim["Driver's_Gender"].replace(np.nan,'J', inplace = True)
pd.unique(claim["Driver's_Gender"])


claim["Driver's_Relationship_with_Insured"].replace(1,'In_person', inplace = True)
claim["Driver's_Relationship_with_Insured"].replace(2,'Relatives', inplace = True)
claim["Driver's_Relationship_with_Insured"].replace(3,'Employee', inplace = True)
claim["Driver's_Relationship_with_Insured"].replace(4,'Vehicle', inplace = True)
claim["Driver's_Relationship_with_Insured"].replace(5,'Others', inplace = True)
claim["Driver's_Relationship_with_Insured"].replace(6,'Spouse', inplace = True)
claim["Driver's_Relationship_with_Insured"].replace(7,'Child', inplace = True)
pd.unique(claim["Driver's_Relationship_with_Insured"])

claim.Marital_Status_of_Driver.replace(1,"Married",inplace = True)
claim.Marital_Status_of_Driver.replace(2,"Unmarried",inplace = True)
pd.unique(claim.Marital_Status_of_Driver)

claim['Claim_Status_(close,_open,_reopen_etc)'].replace(1,'Closed',inplace = True)
claim['Claim_Status_(close,_open,_reopen_etc)'].replace(0,'Yet',inplace = True)
pd.unique(claim['Claim_Status_(close,_open,_reopen_etc)'])

non_claim = (set(le.fit_transform(ALL)[0 : 351273])^set(claim.Policy_Number))&set(le.fit_transform(ALL)[0 : 351273])

nn = pd.DataFrame(np.nan, index=np.arange(len(non_claim)),columns=claim.columns)

nn.Policy_Number = non_claim

cl = claim.append(nn, ignore_index=True)

cl["Nature_of_the_claim"] = cl["Nature_of_the_claim"].fillna("None")
cl["Driver's_Relationship_with_Insured"] = cl["Driver's_Relationship_with_Insured"].fillna("None")
cl["Cause_of_Loss"] = cl["Cause_of_Loss"].fillna("None")
cl["Paid_Loss_Amount"] = cl["Paid_Loss_Amount"].fillna(0)
cl["paid_Expenses_Amount"] = cl["paid_Expenses_Amount"].fillna(0)
cl["Salvage_or_Subrogation?"] = cl["Salvage_or_Subrogation?"].fillna(0)
cl["At_Fault?"] = cl["At_Fault?"].fillna(0)
cl["Claim_Status_(close,_open,_reopen_etc)"] = cl["Claim_Status_(close,_open,_reopen_etc)"].fillna("None")
cl["Deductible"] = cl["Deductible"].fillna(0)
cl["Accident_area"] = cl["Accident_area"].fillna("None")


policy = policy.drop(["Insured's_ID",'Prior_Policy_Number'], axis = 1)

policy.Cancellation.replace(' ','N', inplace = True)
pd.unique(policy.Cancellation)

policy.Imported_or_Domestic_Car.replace(10,'Dom', inplace = True)
policy.Imported_or_Domestic_Car.replace(40,'Jap', inplace = True)
policy.Imported_or_Domestic_Car.replace(24,'A-Jap', inplace = True)
policy.Imported_or_Domestic_Car.replace(50,'Kor', inplace = True)
policy.Imported_or_Domestic_Car.replace(90,'Others', inplace = True)
policy.Imported_or_Domestic_Car.replace(30,'Euro', inplace = True)
policy.Imported_or_Domestic_Car.replace(20,'A', inplace = True)
policy.Imported_or_Domestic_Car.replace(21,'Ford', inplace = True)
policy.Imported_or_Domestic_Car.replace(22,'General', inplace = True)
policy.Imported_or_Domestic_Car.replace(23,'Chrysler', inplace = True)
pd.unique(policy.Imported_or_Domestic_Car)

for i in pd.unique(policy.Coverage_Deductible_if_applied):
    policy.Coverage_Deductible_if_applied.replace(i,eval("'%d'%i"), inplace = True)
pd.unique(policy.Coverage_Deductible_if_applied)

policy.fassured.replace(1,'DN', inplace = True)
policy.fassured.replace(2,'DJ', inplace = True)
policy.fassured.replace(3,'ON', inplace = True)
policy.fassured.replace(4,'OJ', inplace = True)
policy.fassured.replace(5,'Error-N', inplace = True)
policy.fassured.replace(6,'Error-J', inplace = True)
pd.unique(policy.fassured)

policy.fsex.replace('1','M', inplace = True)
policy.fsex.replace('2','F', inplace = True)
policy.fsex.replace(' ','J', inplace = True)
pd.unique(policy.fsex)

policy.fmarriage.replace('1','Married', inplace = True)
policy.fmarriage.replace('2','Unmarried', inplace = True)
policy.fmarriage.replace(' ',np.nan, inplace = True)
pd.unique(policy.fmarriage)

policy.fequipment1.replace(1,'Y', inplace = True)
policy.fequipment1.replace(0,'N', inplace = True)
pd.unique(policy.fequipment1)

policy.fequipment2.replace(1,'Y', inplace = True)
policy.fequipment2.replace(0,'N', inplace = True)
pd.unique(policy.fequipment2)

policy.fequipment3.replace(1,'Y', inplace = True)
policy.fequipment3.replace(0,'N', inplace = True)
pd.unique(policy.fequipment3)

policy.fequipment4.replace(1,'Y', inplace = True)
policy.fequipment4.replace(0,'N', inplace = True)
pd.unique(policy.fequipment4)

policy.fequipment5.replace(1,'Y', inplace = True)
policy.fequipment5.replace(0,'N', inplace = True)
pd.unique(policy.fequipment5)

policy.fequipment6.replace(1,'Y', inplace = True)
policy.fequipment6.replace(0,'N', inplace = True)
pd.unique(policy.fequipment6)

policy.fequipment9.replace(1,'Y', inplace = True)
policy.fequipment9.replace(0,'N', inplace = True)
pd.unique(policy.fequipment9)

policy.Main_Insurance_Coverage_Group.replace('車責','R', inplace = True)
policy.Main_Insurance_Coverage_Group.replace('竊盜','S', inplace = True)
policy.Main_Insurance_Coverage_Group.replace('車損','D', inplace = True)
policy.nequipment9.replace('原裝車含配備','Original', inplace = True)
policy.nequipment9.replace('5合1影音','5in1', inplace = True)
policy.nequipment9.replace('大包','Big', inplace = True)
policy.nequipment9.replace('伸尾                                                                                                ','Tail', inplace = True)
pd.unique(policy.nequipment9)

pl = policy


pl.Replacement_cost_of_insured_vehicle[pl.Replacement_cost_of_insured_vehicle > 5000] = np.nan
pl.pdmg_acc[pl.pdmg_acc > 3] = np.nan
pl.Manafactured_Year_and_Month[pl.Manafactured_Year_and_Month > 50] = np.nan
pl.dbirth[pl.dbirth < 0] = np.nan



train.to_csv('..../training-adj.csv', index = False)
test.to_csv('..../testing-adj.csv', index = False)
cl.to_csv('..../claim_adj.csv', index = False)
pl.to_csv('..../policy_adj.csv', index = False)  
