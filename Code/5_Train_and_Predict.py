import numpy as np
import pandas as pd
import time 
import math
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold
import lightgbm as lgb 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, Imputer
imputer = Imputer(strategy = 'median')
from sklearn.metrics import confusion_matrix

trainset = pd.read_csv("C:/Users/a0972/Desktop/susu/training-adj.csv") # 210763 #
testset = pd.read_csv("C:/Users/a0972/Desktop/susu/testing-adj.csv")
tt = pd.read_csv("C:/Users/a0972/Desktop/T-Brain/testing-set.csv")

data1 = pd.read_csv("C:/Users/a0972/Desktop/susu/dataset1.csv")
data2 = pd.read_csv("C:/Users/a0972/Desktop/susu/dataset2.csv")
data3 = pd.read_csv("C:/Users/a0972/Desktop/susu/dataset3.csv")


def minmaxscale(df):
    ncol = df.shape[1]
    nrow = df.shape[0]
    a = np.zeros((nrow, ncol))
    for i in range(0,ncol) :
        k = df.columns[i]
        a[:, i] = ( df[k] - df[k].min() ) / ( df[k].max() - df[k].min() )
    return a


full = pd.concat([data1, data2, data3], axis = 1)

imputer.fit(full)
full = pd.DataFrame(imputer.transform(full), columns = full.columns)

#########################################
# Predict renewwal clients' Next Premium
#########################################

train = full[0:210763]
test = full[210763:]


Growth_Rate = (trainset.Next_Premium - trainset.Previous_Premium)/(trainset.Previous_Premium)


def outliers_threshold(ys, threshold):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    upper_bound = quartile_3 + (iqr * threshold)
    return upper_bound

    
sp = outliers_threshold(Growth_Rate, 0.5)


tmae = Growth_Rate

start = time.time()

for j in range(0, 10):

    folds = 5
    fpred1 = []
    lgbreg = []
    
    start = time.time()
    
    kf = KFold(train.shape[0], n_folds=folds)
    for i, (train_index, test_index) in enumerate(kf):
        print('\n Fold %d\n' % (i + 1))
        X_train, X_val = train.iloc[train_index,:], train.iloc[test_index,:]
        y_train1, y_val1 = tmae[train_index], tmae[test_index]
    
    #######################################
    #
    # Define cross-validation variables
    #
    #######################################
    
        param1={
        'boosting_type': 'gbdt',
        'class_weight': None,
        'colsample_bytree': 0.733333,
        'learning_rate': 0.00764107,
        'max_depth': -1,
        'min_child_samples': 460,
        'min_child_weight': 0.001,
        'min_split_gain': 0.0,
        'n_estimators': 3000,
        'n_jobs': -1,
        'num_leaves': 77,
        'objective':None,
        'random_state': 42,
        'reg_alpha': 0.877551,
        'reg_lambda': 0.204082,
        'silent': True,
        'subsample': 0.949495,
        'subsample_for_bin': 240000,
        'subsample_freq': 1,
        'metric': 'l1' 
        }
        
        X_train = pd.DataFrame(np.array(X_train[y_train1 != -1]), columns = train.columns)
        y_train1 = np.array(y_train1[y_train1 != -1])
        
        X_train = pd.DataFrame(np.array(X_train[y_train1 <= sp]), columns = train.columns)
        y_train1 = np.array(y_train1[y_train1 <= sp])
    
        train_data1 = lgb.Dataset(X_train,label=y_train1)
        valid_data1 = lgb.Dataset(X_val,label = y_val1)
       
    ####################################
    #  Build Model
    ####################################
       
        lgbm_reg = lgb.train(param1,
        train_data1,
        2500,
        valid_sets=valid_data1,
        early_stopping_rounds= 40,
        verbose_eval= 100
        )
        
    ####################################
    #  Evaluate Model and Predict
    ####################################
        
        lgbreg = lgbreg + list(lgbm_reg.predict(X_val))
        y_pred1 = lgbm_reg.predict(test)
    
    
    ####################################
    #  feature importance
    ####################################
    
        feature_importances_reg = pd.DataFrame({'feature': train.columns, 'importance': lgbm_reg.feature_importance()})
     
        zero_importance_features_reg = list(feature_importances_reg.feature[feature_importances_reg.importance == 0])
          
    ####################################
    #  Add Predictions and Average Them
    ####################################
    
        if i > 0:
            fpred1 = pred1 + y_pred1
            useless = list(set(useless) & set(zero_importance_features_reg) )
        else:
            fpred1 = y_pred1
            useless = zero_importance_features_reg
        pred1 = fpred1
    
    zero_importance_features_reg_union = useless
    
    
    if j == 0 :
        pred_reg = lgbreg
        mpred1 = pred1 / folds
        train = train.drop(labels = zero_importance_features_reg_union, axis = 1)
        test = test.drop(labels = zero_importance_features_reg_union, axis = 1)
    else :
        if mean_absolute_error(trainset.Next_Premium, (1 + np.array(lgbreg) ) * trainset.Previous_Premium) >= mean_absolute_error(trainset.Next_Premium, (1 + np.array(pred_reg) ) * trainset.Previous_Premium) :
            break
        else :
            pred_reg = lgbreg
            mpred1 = pred1 / folds
            train = train.drop(labels = zero_importance_features_reg_union, axis = 1)
            test = test.drop(labels = zero_importance_features_reg_union, axis = 1)
            
    




train = full[0:210763]
test = full[210763:]

trainset['renewal'] = trainset.Next_Premium
trainset.renewal[trainset.renewal > 0] = 1


##########################
# Predict renewwal or not
##########################

tauc = trainset.renewal


for j in range(0, 10):

    folds = 5
    fpred2 = []
    lgbclf = []
    
    kf = KFold(train.shape[0], n_folds=folds)
    for i, (train_index, test_index) in enumerate(kf):
        print('\n Fold %d\n' % (i + 1))
        X_train, X_val = train.iloc[train_index,:], train.iloc[test_index,:]
        y_train2, y_val2 = tauc[train_index], tauc[test_index]
        
    #######################################
    #
    # Define cross-validation variables
    #
    #######################################
        
        param2={
        'boosting_type': 'gbdt',
        'class_weight': None,
        'colsample_bytree': 0.733333,
        'learning_rate': 0.00764107,
        'max_depth': -1,
        'min_child_samples': 460,
        'min_child_weight': 0.001,
        'min_split_gain': 0.0,
        'n_estimators': 3000,
        'n_jobs': -1,
        'num_leaves': 20,
        'objective':'binary',
        'random_state': 42,
        'reg_alpha': 0.877551,
        'reg_lambda': 0.204082,
        'silent': True,
        'subsample': 0.949495,
        'subsample_for_bin': 240000,
        'subsample_freq': 1,
        'metric': 'auc'
        } 
            
        train_data2 = lgb.Dataset(X_train,label=y_train2)
        valid_data2 = lgb.Dataset(X_val,label = y_val2)
        
    ####################################
    #  Build Model
    ####################################
    
        lgbm_clf = lgb.train(param2,
        train_data2,
        2500,
        valid_sets=valid_data2,
        early_stopping_rounds= 40,
        verbose_eval= 100
        )
        
    ####################################
    #  Evaluate Model and Predict
    ####################################
        
        lgbclf  = lgbclf  + list(lgbm_clf.predict(X_val))
        y_pred2 = lgbm_clf.predict(test)
    
    ####################################
    #  feature importance
    ####################################
    
        feature_importances_clf = pd.DataFrame({'feature': train.columns, 'importance': lgbm_clf.feature_importance()})
        
        zero_importance_features_clf = list(feature_importances_clf.feature[feature_importances_clf.importance == 0])
     
    ####################################
    #  Add Predictions and Average Them
    ####################################
    
        if i > 0:
            fpred2 = pred2 + y_pred2
            useless = list(set(useless) & set(zero_importance_features_clf) )
        else:
            fpred2 = y_pred2
            useless = zero_importance_features_clf
        pred2 = fpred2
    
    zero_importance_features_clf_union = useless
    
    if j == 0 :
        pred_clf = lgbclf
        mpred2 = pred2 / folds
        train = train.drop(labels = zero_importance_features_clf_union, axis = 1)
        test = test.drop(labels = zero_importance_features_clf_union, axis = 1)
    else :
        if roc_auc_score(tauc, lgbclf) <= roc_auc_score(tauc, pred_clf) :
            break
        else :
            pred_clf = lgbclf
            mpred2 = pred2 / folds
            train = train.drop(labels = zero_importance_features_clf_union, axis = 1)
            test = test.drop(labels = zero_importance_features_clf_union, axis = 1)


end = time.time()
print("Running Time : %d min %d sec"%(math.floor((end - start) / 60),math.ceil((end - start) % 60 )))


j = 0
acu = [0] * len(np.arange(0,1,0.01))
for i in np.arange(0,1,0.01):
    pred = np.array(pred_clf)
    pred[pred >= i] = 1
    pred[pred < i] = 0
    acu[j] = accuracy_score(tauc,pred)
    j =j + 1

threshold = np.arange(0,1,0.01)[acu == max(acu)][0]




## Validate set's MAE ##

val_pred = (1 + np.array(pred_reg) ) * trainset.Previous_Premium

val_pred[np.array(pred_clf) < threshold] = 0


mean_absolute_error(trainset.Next_Premium, val_pred)

########################


predict = (1 + np.array(mpred1) ) * testset.Previous_Premium 

predict[mpred2 < threshold] = 0

submit = tt[['Policy_Number']]
submit['Next_Premium'] = predict


submit.to_csv('..../submit.csv', index = False)







