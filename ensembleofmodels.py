import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
#######################################################################################
### This python code assigns weights for models devloped by advanced regression techniques (lasso,ridge,elasticnet,xgboost) used in ensemble methods

#loading training data & test data
trai_data = pd.read_csv('Downloads/train.csv')
test_data = pd.read_csv('Downloads/test.csv')

###################### PREPROCESSING ######################################
#removing outliers in data
trai_data.drop(trai_data.index[[523,1298,30,495,968]], inplace=True)

#applying logarthmic transformation to Response variable : SalePrice
target = trai_data['SalePrice'][trai_data["SalePrice"] != 0].apply(np.log)

#removing SalePrice & openporch+wooddeck(added during preprocessing
del trai_data['SalePrice']
del trai_data['openporch+wooddeck']


#Combining training and test data for performing preprocessing
train_data = pd.concat((trai_data, test_data), ignore_index=True)


#selecting numeric features and replacing missing values by mean
numeric_feats = train_data.dtypes[train_data.dtypes != "object"].index
for i in numeric_feats:
    train_data[i]=train_data[i].fillna(train_data[i].mean())
    train_data[i][train_data[i] == 0] = 1
    

#applying logarthmic transformation to numeric features and storing in x_data
x_data = pd.DataFrame(train_data[numeric_feats].apply(np.log))


#selecting categorical features and converting to dummy variables(one hot encoding)
categoric_feats = train_data.dtypes[train_data.dtypes == "object"].index
y_data = pd.get_dummies(train_data[categoric_feats])

#Combining data after preprocessing steps
train = (x_data.join(y_data))

#taking test data from combined data
#test_data = train[1456:]


#Taking training data from combined data
X=pd.DataFrame(train.head(n=1455))


#combining training data with response variable
Y=X.join(target)

fig, (axis1) = plt.subplots(1,1,figsize=(15,4))
Y.plot(kind='scatter',x='GrLivArea',y='SalePrice',figsize=(15,4))
fig, (axis1) = plt.subplots(1,1,figsize=(15,4))

plt.show()

############# Saving preprocessed data for ensemble methods #########

#X.to_csv("F:/Books/projectX/ridge_train.csv", index = False)
#X.to_csv("F:/Books/projectX/lasso_train.csv", index = False)
#X.to_csv("F:/Books/projectX/elastic_train.csv", index = False)

#test.to_csv("F:/Books/projectX/ridge_test.csv", index = False)
#test.to_csv("F:/Books/projectX/lasso_test.csv", index = False)
#test.to_csv("F:/Books/projectX/elastic_test.csv", index = False)

########## DEVELOPING PREDICTIVE MODELS (LASSO,RIDGE,ELASTIC AND XGBOOST)###############################

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

#Lasso Model
model_lasso = LassoCV(alphas = [ 0.0005],cv = 5).fit(X[0:], target[0:])

lasso_preds = (model_lasso.predict(X[1000:]))
lasso_preds_test = (model_lasso.predict(train[1455:]))

#Elastic model

model_lastic1 = ElasticNet(alpha=0.001, l1_ratio=0.288, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, 
copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic').fit(X[0:1000],target[0:1000])

elastic_preds = (model_lastic1.predict(X[1000:]))

#Ridge model

model_lastic1 = ElasticNet(alpha=0.001, l1_ratio=0, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, 
copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic').fit(X[0:],target[0:])

ridge_preds = (model_lastic1.predict(X[1000:]))
ridge_preds_test = (model_lastic1.predict(train[1455:]))



######################
#Selecting weights for different predictive models in ensemble method

#Combining predictions from different models
boruta_preds = pd.read_csv('F:/Books/projectX/boruta_results.csv')
#xg_preds = pd.DataFrame({"p1": xg_preds})
lasso_preds = pd.DataFrame({"p2":lasso_preds})
ridge_preds = pd.DataFrame({"p3":ridge_preds})
elastic_preds = pd.DataFrame({"p4":elastic_preds})

ensemble_data2 = ridge_preds.join(lasso_preds)
ensemble_data3 = ensemble_data2.join(elastic_preds)
ensemble_data_final = ensemble_data3.join(boruta_preds)


#Taking target variable from training dataset
input_data = pd.read_csv('Downloads/train.csv')

input_data.drop(input_data.index[[523,1298,30,495,968]], inplace=True)
target1 = input_data['SalePrice'][input_data["SalePrice"] != 0].apply(np.log)



#print(ensemble_data_final.corr(method='pearson', min_periods=1))

############## Tuning parameters of lasso model which is used for assigning weights to ridge, lasso, elastic, baruto techniques.

model_lasso1 = LassoCV(alphas = [1e-15, 1e-6, 1e-7,1e-3,1e-4,1e-8,1, 0.1, 0.001, 0.0005,0.5,0.1,1],cv = 5).fit(ensemble_data_final[0:], target1[1000:])

### displaying weights

print(model_lasso1.mse_path_)
coef = pd.Series(model_lasso1.coef_, index = ensemble_data_final.columns)
print(model_lasso1.coef_)
print(model_lasso1.alpha_)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


#Lasso picked 3 variables and eliminated the other 1 variables




############################################################



#Test data predictions


"""
preds = np.expm1((0.0718859 * xg_test) + (0.31312871 * ridge_preds_test) + (0.63698145 * lasso_preds_test))
solution = pd.DataFrame({"Id":test_data.Id, "SalePrice":preds})
solution.to_csv("F:/Books/projectX/ensemble_sol.csv", index = False)
"""





