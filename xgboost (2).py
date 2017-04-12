
#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

#train = pd.read_csv('train_modified.csv')

#IDcol = 'ID'



#Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
import seaborn as sns 
#Reading Data
train_data = pd.read_csv('Downloads/train.csv')
test_data = pd.read_csv('Downloads/test.csv')


#Outlier removal

df1 = (train_data.ix[((train_data["GrLivArea"]>8.3) & (train_data["SalePrice"]<12.5))])
df2 = (train_data.ix[((train_data["GrLivArea"]>6.5) & (train_data["SalePrice"]<10.7))])
print(df2)

train_data.drop(train_data.index[[523,1298,30,495,968]], inplace=True)
train_data.info()

fig, (axis1) = plt.subplots(1,1,figsize=(15,4))
train_data.plot(kind='scatter',x='GrLivArea',y='SalePrice',figsize=(15,4))
fig, (axis1) = plt.subplots(1,1,figsize=(15,4))

train_data["GrLivArea"] = train_data['GrLivArea'][train_data["GrLivArea"] != 0].apply(np.log)
train_data["SalePrice"] = train_data['SalePrice'][train_data["SalePrice"] != 0].apply(np.log)

target = 'SalePrice'

train_data.plot(kind='scatter',x='GrLivArea',y='SalePrice',figsize=(15,4))


fig, (axis1) = plt.subplots(1,1,figsize=(15,4))
train_data.plot(kind='scatter',x='SalePrice',y='SalePrice',figsize=(15,4))
fig, (axis1) = plt.subplots(1,1,figsize=(15,4))



"""sns.boxplot( y="SalePrice", data=train_data)
sns.stripplot( y="SalePrice", data=train_data,
              size=4, jitter=True, edgecolor="gray")
              """
              
plt.show()
#Replacing missing data with MEDIAN
#train_data["GarageArea"][train_data["GarageArea"] != train_data["GarageArea"]] = train_data["GarageArea"].median()
#Information gathering
#print(train_data.info())
#CountPlots
#fig, (axis1) = plt.subplots(1,1,figsize=(15,4))
#sns.countplot(x='Electrical', data=train_data,palette="husl", ax=axis1)

#Finding Correlation between Variables
#print(train_data.corr(method='pearson', min_periods=1))
print(train_data['GrLivArea'].corr(train_data['MasVnrArea']))


#Applying min_max Scaler
min_max_scaler = preprocessing.MinMaxScaler()
#train_data["GarageArea"]= pd.DataFrame(data = min_max_scaler.fit_transform(train_data["GarageArea"]))

#train_data["SalePrice"] = train_data['SalePrice'][train_data["SalePrice"] != 0].apply(np.log)

#Taking Response variable in 'Y'
Y = train_data["SalePrice"]
#del train_data["SalePrice"]

#Taking Log on continous values
train_data["LotArea"] = train_data['LotArea'][train_data["LotArea"] != 0].apply(np.log)
train_data["YearBuilt"] = train_data['YearBuilt'][train_data["YearBuilt"] != 0].apply(np.log)
#train_data["GrLivArea"] = train_data['GrLivArea'][train_data["GrLivArea"] != 0].apply(np.log)



test_data["LotArea"] = test_data['LotArea'][test_data["LotArea"] != 0].apply(np.log)
test_data["YearBuilt"] = test_data['YearBuilt'][test_data["YearBuilt"] != 0].apply(np.log)
test_data["GrLivArea"] = test_data['GrLivArea'][test_data["GrLivArea"] != 0].apply(np.log)

#Adding up openporch+wooddeck to understand them together
#train_data["GarageArea"] = train_data['GarageArea'][train_data["GarageArea"] != 0].apply(np.log)

#creating new DataFrame - train
t_data = pd.DataFrame(train_data["LotArea"])
t_data["Neighborhood"] = train_data["Neighborhood"]
t_data["YearBuilt"] = train_data["YearBuilt"]
t_data["ExterQual"] = train_data["ExterQual"]
t_data["GrLivArea"] = train_data["GrLivArea"]
t_data["SalePrice"] = train_data["SalePrice"]

tst_data = pd.DataFrame(test_data["LotArea"])
tst_data["Neighborhood"] = test_data["Neighborhood"]
tst_data["YearBuilt"] = test_data["YearBuilt"]
tst_data["ExterQual"] = test_data["ExterQual"]
tst_data["GrLivArea"] = test_data["GrLivArea"]


#Categorical variable to dummy variable conversion
t_data = t_data.join(pd.get_dummies(t_data['Neighborhood'], prefix="Neighborhood"))
del t_data["Neighborhood"]

t_data = t_data.join(pd.get_dummies(t_data['ExterQual'], prefix="ExterQual"))
del t_data["ExterQual"]

t_data.to_csv("F:/Books/projectX/ensemble/xgboosttrain.csv", index = False)
#t_data = t_data.join(pd.get_dummies(t_data['Foundation'], prefix="Foundation"))
#del t_data["Foundation"]

#t_data= pd.DataFrame(data = min_max_scaler.fit_transform(t_data), columns = t_data.columns, index=t_data.index)

tst_data = tst_data.join(pd.get_dummies(tst_data['Neighborhood'], prefix="Neighborhood"))
del tst_data["Neighborhood"]

tst_data = tst_data.join(pd.get_dummies(tst_data['ExterQual'], prefix="ExterQual"))
del tst_data["ExterQual"]

#tst_data.to_csv("F:/Books/projectX/ensemble/xgboost.csv", index = False)

from sklearn.metrics import mean_squared_error


def modelfit(alg, dtrain,dtest, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=100):
    
    if useTrainCV:
        
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        #cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=500, nfold=cv_folds,
         #   metrics='rmse', early_stopping_rounds=early_stopping_rounds)
        #alg.set_params(n_estimators=cvresult.shape[0])
        
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['SalePrice'],eval_metric='rmse')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtest_predictions = alg.predict(dtest[predictors])
    #dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    print(pd.DataFrame(dtest_predictions).head())
    solution = pd.DataFrame({"Id":test_data.Id, "SalePrice":dtest_predictions})
    solution.to_csv("F:/Books/projectX/xgboost_preds.csv", index = False)
        
    #Print model report:
    print "\nModel Report"
    rmse= np.sqrt(mean_squared_error(Y[0:],dtrain_predictions))
    print(rmse)
    #print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    #print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['SalePrice'], dtrain_predprob)



    
    
    
    
    
    
    
    
    
    #Choose all predictors except target & IDcols


predictors = [x for x in t_data.columns if x not in [target]]
xgb1 = XGBRegressor(
 learning_rate =0.16,
 n_estimators=114,
 max_depth=3,
 min_child_weight=3,
 gamma=0.02,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.01,
 objective= 'reg:linear',
 nthread=4)
modelfit(xgb1, t_data,tst_data, predictors)


"""
predictors = [x for x in t_data.columns if x not in [target]]

param_test1 = {
 'n_estimators':range(90,105,1),
}


gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='mean_squared_error', cv=5)
result = gsearch1.fit(t_data[predictors],t_data[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
print("Best: %f using %s" % (result.best_score_, result.best_params_))

means, stdevs = [], []
for params, mean_score, scores in result.grid_scores_:
	stdev = scores.std()
	means.append(mean_score)
	stdevs.append(stdev)
	print("%f (%f) with: %r" % (np.sqrt(-1*mean_score), stdev, params))
	
	

#n_estimators : 102  && 0.160206

"""
"""
predictors = [x for x in t_data.columns if x not in [target]]

param_test1 = {
 'max_depth':range(2,7,1),
}


gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=102,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='mean_squared_error', cv=5)
result = gsearch1.fit(t_data[predictors],t_data[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
print("Best: %f using %s" % (np.sqrt(-1*result.best_score_), result.best_params_))

means, stdevs = [], []
for params, mean_score, scores in result.grid_scores_:
	stdev = scores.std()
	means.append(mean_score)
	stdevs.append(stdev)
	print("%f (%f) with: %r" % (np.sqrt(-1*mean_score), stdev, params))

"""
#Best: 0.160033 using {'max_depth': 4}

"""
predictors = [x for x in t_data.columns if x not in [target]]



n_estimators = range(90,105,1)
max_depth = [2, 3, 4, 5,6]
print(max_depth)
param_test1 = dict(max_depth=max_depth, n_estimators=n_estimators)


gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=102,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='mean_squared_error', cv=5)
result = gsearch1.fit(t_data[predictors],t_data[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
print("Best: %f using %s" % (np.sqrt(-1*result.best_score_), result.best_params_))

means, stdevs = [], []
for params, mean_score, scores in result.grid_scores_:
	stdev = scores.std()
	means.append(mean_score)
	stdevs.append(stdev)
	print("%f (%f) with: %r" % (np.sqrt(-1*mean_score), stdev, params))

# Best: 0.160033 using {'n_estimators': 102, 'max_depth': 4}

"""


"""
predictors = [x for x in t_data.columns if x not in [target]]

param_test1 = {
 'learning_rate':np.arange(0.10,0.18,0.01),
}



gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=102,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='mean_squared_error', cv=5)
result = gsearch1.fit(t_data[predictors],t_data[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
print("Best: %f using %s" % (np.sqrt(-1*result.best_score_), result.best_params_))

means, stdevs = [], []
for params, mean_score, scores in result.grid_scores_:
	stdev = scores.std()
	means.append(mean_score)
	stdevs.append(stdev)
	print("%f (%f) with: %r" % (np.sqrt(-1*mean_score), stdev, params))

# Best: 0.158939 using {'learning_rate': 0.13}

"""

"""
predictors = [x for x in t_data.columns if x not in [target]]



min_child_weight = range(3,10,1)
max_depth = [2, 3, 4, 5,6]
n_estimators = range(100,110,1)
print(max_depth)
param_test1 = dict(max_depth=max_depth, min_child_weight=min_child_weight,n_estimators=n_estimators)


gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.13, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='mean_squared_error', cv=5)
result = gsearch1.fit(t_data[predictors],t_data[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
print("Best: %f using %s" % (np.sqrt(-1*result.best_score_), result.best_params_))

means, stdevs = [], []
for params, mean_score, scores in result.grid_scores_:
	stdev = scores.std()
	means.append(mean_score)
	stdevs.append(stdev)
	print("%f (%f) with: %r" % (np.sqrt(-1*mean_score), stdev, params))

#Best: 0.158559 using {'n_estimators': 107, 'max_depth': 3, 'min_child_weight': 3}

"""
"""
predictors = [x for x in t_data.columns if x not in [target]]

param_test1 = {
 'gamma':np.arange(0,0.15,0.01),
}



gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.13, n_estimators=107,max_depth=3,
 min_child_weight=3,subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='mean_squared_error', cv=5)
result = gsearch1.fit(t_data[predictors],t_data[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
print("Best: %f using %s" % (np.sqrt(-1*result.best_score_), result.best_params_))

means, stdevs = [], []
for params, mean_score, scores in result.grid_scores_:
	stdev = scores.std()
	means.append(mean_score)
	stdevs.append(stdev)
	print("%f (%f) with: %r" % (np.sqrt(-1*mean_score), stdev, params))

# Best: 0.158543 using {'gamma': 0.02}      

"""
"""
predictors = [x for x in t_data.columns if x not in [target]]



subsample = np.arange(0.1,1,0.1)
colsample_bytree =np.arange(0.1,1,0.1)

param_test1 = dict(subsample=subsample, colsample_bytree=colsample_bytree)


gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.13, n_estimators=107,max_depth=3,gamma = 0.02,
 min_child_weight=3,subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='mean_squared_error', cv=5)
result = gsearch1.fit(t_data[predictors],t_data[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
print("Best: %f using %s" % (np.sqrt(-1*result.best_score_), result.best_params_))

means, stdevs = [], []
for params, mean_score, scores in result.grid_scores_:
	stdev = scores.std()
	means.append(mean_score)
	stdevs.append(stdev)
	print("%f (%f) with: %r" % (np.sqrt(-1*mean_score), stdev, params))

#Best: 0.158543 using {'subsample': 0.80000000000000004, 'colsample_bytree': 0.80000000000000004}
"""
"""
predictors = [x for x in t_data.columns if x not in [target]]

param_test1 = {
 'reg_alpha':[1e-5, 1e-6, 1e-7,1e-3,1e-4,1e-8]
}



gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.13, n_estimators=107,max_depth=3,gamma = 0.02,
 min_child_weight=3,subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='mean_squared_error', cv=5)
result = gsearch1.fit(t_data[predictors],t_data[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
print("Best: %f using %s" % (np.sqrt(-1*result.best_score_), result.best_params_))

means, stdevs = [], []
for params, mean_score, scores in result.grid_scores_:
	stdev = scores.std()
	means.append(mean_score)
	stdevs.append(stdev)
	print("%f (%f) with: %r" % (np.sqrt(-1*mean_score), stdev, params))

#Best: 0.158530 using {'reg_alpha': 0.001}
"""
"""
# final chheck of learning rate and number of trees for bias and variance trade off 

predictors = [x for x in t_data.columns if x not in [target]]



learning_rate = np.arange(0.05,0.2,0.01)
n_estimators = range(95,120,1)

param_test1 = dict(learning_rate=learning_rate, n_estimators=n_estimators)


gsearch1 = GridSearchCV(estimator = XGBRegressor(max_depth=3,gamma = 0.02,reg_alpha=0.01,
 min_child_weight=3,subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='mean_squared_error', cv=5)
result = gsearch1.fit(t_data[predictors],t_data[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
print("Best: %f using %s" % (np.sqrt(-1*result.best_score_), result.best_params_))

means, stdevs = [], []
for params, mean_score, scores in result.grid_scores_:
	stdev = scores.std()
	means.append(mean_score)
	stdevs.append(stdev)
	print("%f (%f) with: %r" % (np.sqrt(-1*mean_score), stdev, params))

#Best: 0.157448 using {'n_estimators': 114, 'learning_rate': 0.16000000000000003}
"""
"""
# final tuned parameters 


predictors = [x for x in t_data.columns if x not in [target]]



learning_rate = np.arange(0.05,0.2,0.01)
n_estimators = range(95,120,1)

param_test1 = dict(learning_rate=learning_rate, n_estimators=n_estimators)


gsearch1 = GridSearchCV(estimator = XGBRegressor(n_estimators=114,learning_rate=0.16,max_depth=3,gamma = 0.02,reg_alpha=0.01,
 min_child_weight=3,subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='mean_squared_error', cv=5)
result = gsearch1.fit(t_data[predictors],t_data[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
print("Best: %f using %s" % (np.sqrt(-1*result.best_score_), result.best_params_))

means, stdevs = [], []
for params, mean_score, scores in result.grid_scores_:
	stdev = scores.std()
	means.append(mean_score)
	stdevs.append(stdev)
	print("%f (%f) with: %r" % (np.sqrt(-1*mean_score), stdev, params))
	
"""
