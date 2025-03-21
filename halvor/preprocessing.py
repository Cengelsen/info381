from sklearn.ensemble import RandomForestClassifier
import shap 
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error as MSE
from sklearn import preprocessing 
import datetime as dt

#Gammelt datasett bytt ut!
train_data = pd.read_csv("fraudTrain.csv", sep=',')
test_data = pd.read_csv("fraudTest.csv", sep=',')

label_encoder = preprocessing.LabelEncoder()
train_data['job'] = label_encoder.fit_transform(train_data['job'])
train_data['category'] = label_encoder.fit_transform(train_data['category'])
train_data['gender'] = label_encoder.fit_transform(train_data['gender'])

train_data['dob']= pd.to_datetime(train_data['dob'])
train_data['birth_year'] = train_data['dob'].dt.year

test_data['job'] = label_encoder.fit_transform(test_data['job'])
test_data['category'] = label_encoder.fit_transform(test_data['category'])
test_data['gender'] = label_encoder.fit_transform(test_data['gender'])

test_data['dob']= pd.to_datetime(test_data['dob'])
test_data['birth_year'] = test_data['dob'].dt.year

X_train = train_data.drop(columns=['Unnamed: 0','cc_num','merchant','first','last', 'street', 'city', 'state','lat','long','trans_num','unix_time','dob','trans_date_trans_time', 'is_fraud'])
y_train = train_data['is_fraud']
X_test = test_data.drop(columns=['Unnamed: 0','cc_num','merchant','first','last', 'street', 'city', 'state','lat','long','trans_num','unix_time','dob','trans_date_trans_time', 'is_fraud'])
y_test = test_data['is_fraud']

print(X_train.isnull().sum()) 
print(X_train.dtypes)
print(np.unique(y_train))
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

rfc = RandomForestClassifier(max_depth=2, random_state=42)
rfc.fit(X_train, y_train)

xgb_pred = model.predict(X_test)
rfc_pred = rfc.predict(X_test)

xgb_rmse = np.sqrt(MSE(y_test, xgb_pred))
rfc_rmse = np.sqrt(MSE(y_test, rfc_pred))
print(f"XGBoost RMSE: {xgb_rmse:.2f}")
print(f"Random Forest RMSE: {rfc_rmse:.2f}")


xgb_explainer = shap.TreeExplainer(model)
rfc_explainer = shap.TreeExplainer(rfc)
xgb_shap_values = xgb_explainer(X_test)
rfc_shap_values = rfc_explainer(X_test)

print(type(xgb_shap_values))
print(np.shape(xgb_shap_values))

shap.summary_plot(xgb_shap_values,X_test)

print(type(rfc_shap_values))
print(np.shape(rfc_shap_values))

shap.summary_plot(rfc_shap_values,X_test)
