from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from preprocessing import clean_data
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import time
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import shap

data = clean_data("fraud.csv")

# additional preprocessing for neural network


data["amt"] = np.log(data["amt"])
data["city_pop"] = np.log(data["city_pop"])

print("Cyclically encoding features...")
data["trans_minute_sin"] = np.sin(2*np.pi *data["trans_minute"]/60)
data["trans_minute_cos"] = np.cos(2*np.pi *data["trans_minute"]/60)
data["trans_hour_sin"] = np.sin(2*np.pi *data["trans_hour"]/24)
data["trans_hour_cos"] = np.cos(2*np.pi *data["trans_hour"]/24)
data["trans_day_sin"] = np.sin(2*np.pi *data["trans_day"]/31)
data["trans_day_cos"] = np.cos(2*np.pi *data["trans_day"]/31)
data["trans_month_sin"] = np.sin(2*np.pi *data["trans_month"]/12)
data["trans_month_cos"] = np.cos(2*np.pi *data["trans_month"]/12)
data["trans_dayofweek_sin"] = np.sin(2*np.pi *data["trans_dayofweek"]/7)
data["trans_dayofweek_cos"] = np.cos(2*np.pi *data["trans_dayofweek"]/7)
data["dob_day_sin"] = np.sin(2*np.pi *data["dob_day"]/31)
data["dob_day_cos"] = np.cos(2*np.pi *data["dob_day"]/31)
data["dob_month_sin"] = np.sin(2*np.pi *data["dob_month"]/12)
data["dob_month_cos"] = np.cos(2*np.pi *data["dob_month"]/12)

# normalize data

tobnormalized = ['cc_num', 'merchant', 'category','gender', 'street', 'city',
       'state', 'zip', 'job', 'trans_num',
       'merchant_buyer_distance', 'merchant_cluster_id', 'buyer_cluster_id']

scaler = StandardScaler()

print("Scaling features...")
for col in tobnormalized:
    data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1)).flatten() 


X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

param_grid = {
    "classifier__hidden_layer_sizes": [(5, 2)],
    "classifier__solver": ["adam"],
    "classifier__max_iter": [200],
    "classifier__random_state": [42],
    "classifier__activation": ["relu"],
}

time_start = time.time()

# Combine SMOTE and undersampling
over = SMOTE(sampling_strategy=0.1)  # Increase minority class to 10% of majority
under = RandomUnderSampler(sampling_strategy=0.5)  # Reduce majority class

steps = [('over', over), ('under', under), ('classifier', MLPClassifier())]
pipeline = Pipeline(steps=steps)

# Use pipeline in GridSearchCV instead
grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='f1')

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

best_pipeline = grid_search.best_estimator_

print(best_pipeline)

print(f"Time taken for grid search: {time.time() - time_start:.2f} seconds")

time_start = time.time()

y_pred = best_pipeline.predict(X_test)

print(f"Time taken for training: {time.time() - time_start:.2f} seconds")

"""
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Fraud", "Fraud"])
disp.plot(cmap="Oranges")
plt.show()

# Precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall curve')
plt.show()
"""

clf = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=42, activation="relu", max_iter=200, solver="adam")
clf.fit(X_train, y_train)

# Create a small background dataset (subset of training data)
background = shap.sample(X_train, 100)  # Use 100 samples for background

# Create the explainer with your model's predict function
explainer = shap.KernelExplainer(clf.predict_proba, background)

# Calculate SHAP values (this may take some time)
shap_values = explainer.shap_values(X_test[:10000])

# The rest of your code remains the same
if isinstance(shap_values, list):
    global_shap = np.mean(np.abs(np.vstack(shap_values)), axis=0)
else:
    global_shap = np.mean(np.abs(shap_values), axis=0)

# Visualize - for binary classification, index 1 is for positive class
shap.summary_plot(shap_values, X_test[:10000], feature_names=X.columns)


""" shap.waterfall_plot(explainer(X_test[0]))

shap.plots.beeswarm(explainer(X_test[:10000]))

shap.plots.bar(explainer(X_test[:10000]))

shap.plots.heatmap(explainer(X_test[:10000]))

shap.plots.scatter(explainer(X_test[:10000]))

shap.plots.scatter(explainer(X_test[:10000]), color=explainer(X_test[:10000]))
 """
""" shap_values = explanation.values
# make sure the SHAP values add up to marginal predictions
np.abs(shap_values.sum(axis=1) + explanation.base_values - y_pred).max() """