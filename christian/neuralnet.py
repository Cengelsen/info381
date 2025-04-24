from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from preprocessing import clean_data
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import shap
import tensorflow as tf
import shap
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import random

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


data = data.sample(frac=1).reset_index(drop=True)

X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

feature_names = X_train.columns.tolist()

# Create and train neural network
nn = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=X_train.shape[1:]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

nn.compile(optimizer='adam',
           loss='binary_crossentropy',
           metrics=['accuracy'])

nn.fit(X_train, y_train, epochs=10, batch_size=32)



# Sample test data (limit to a reasonable number to prevent memory issues)
background = X_test.sample(min(1000, len(X_test)))
print(f"Using {len(background)} samples for SHAP analysis")

# Initialize SHAP explainer with TensorFlow model
print("Initializing SHAP DeepExplainer...")
explainer = shap.DeepExplainer(nn, background)

# Calculate SHAP values
print("Calculating SHAP values...")
shap_values = explainer.shap_values(X_test.values)


print(f"SHAP values type: {type(shap_values)}")
print(f"Length of SHAP list: {len(shap_values)}")
print(f"Shape of first element: {np.array(shap_values[0]).shape}")
print(f"Shape of second element: {np.array(shap_values[1]).shape}")


#----------------------------------------------------------------------------------
# PLOTS OF GLOBAL SHAPLEY VALUES
#----------------------------------------------------------------------------------

# After calculating SHAP values
print("Reshaping SHAP values for plotting...")

# Check the actual structure of the returned SHAP values
if isinstance(shap_values, list):
    # If DeepExplainer returned a list of arrays (one per class)
    shap_values_class_0 = shap_values[0]
    shap_values_class_1 = shap_values[1]
else:
    # If DeepExplainer returned a single array
    # For binary classification with a sigmoid output, the SHAP values
    # represent the positive class (fraud)
    shap_values_class_1 = shap_values
    # For the negative class, we can take the negative of the SHAP values
    shap_values_class_0 = -shap_values

# Ensure proper shape: should match X_test's shape
if shap_values_class_0.shape != X_test.shape:
    shap_values_class_0 = np.squeeze(shap_values_class_0)
if shap_values_class_1.shape != X_test.shape:
    shap_values_class_1 = np.squeeze(shap_values_class_1)

# Generate summary plots
print("\nGenerating summary plot for non-fraud (class 0)...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values_class_0, X_test, feature_names=feature_names, 
                 max_display=10, show=False, plot_type="bar")
plt.tight_layout()
plt.savefig("shap_summary_nonfraud.png")
plt.close()

print("\nGenerating summary plot for fraud (class 1)...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values_class_1, X_test, feature_names=feature_names, 
                 max_display=10, show=False)
plt.tight_layout()
plt.savefig("shap_summary_fraud.png")
plt.close()

# Generate beeswarm plots
print("\nGenerating beeswarm plot for non-fraud (class 0)...")
plt.figure(figsize=(12, 10))
explanation_nonfraud = shap.Explanation(
    values=shap_values_class_0,
    base_values=np.repeat(explainer.expected_value[0] if isinstance(explainer.expected_value, list) 
                          else explainer.expected_value, X_test.shape[0]),
    data=X_test.values,
    feature_names=feature_names
)
shap.plots.beeswarm(explanation_nonfraud, max_display=10, show=False)
plt.tight_layout()
plt.savefig("shap_beeswarm_nonfraud.png")
plt.close()

print("\nGenerating beeswarm plot for fraud (class 1)...")
plt.figure(figsize=(12, 10))
explanation_fraud = shap.Explanation(
    values=shap_values_class_1,
    base_values=np.repeat(explainer.expected_value[1] if isinstance(explainer.expected_value, list) 
                         else explainer.expected_value, X_test.shape[0]),
    data=X_test.values,
    feature_names=feature_names
)
shap.plots.beeswarm(explanation_fraud, max_display=10, show=False)
plt.tight_layout()
plt.savefig("shap_beeswarm_fraud.png")
plt.close()

print("All SHAP plots saved to disk.")


""" 
shap_values_class_1 = np.squeeze(shap_values)  # Remove singleton dimension

print("\nGenerating plots for class 1 (fraud)...")
print("\nSummary plot")
shap.summary_plot(
    shap_values_class_1, 
    X_test, 
    feature_names=feature_names, 
    max_display=10, 
    plot_size=(12, 10)
)

explanation = shap.Explanation(
    values=shap_values_class_1,
    base_values=np.repeat(explainer.expected_value[0], X_test.shape[0]),
    data=X_test.values,
    feature_names=feature_names
)

print("\nBeeswarm plot")
shap.plots.beeswarm(
    explanation, 
    max_display=10, 
    plot_size=(12, 10)
)

print("\nGenerating plots for class 0 (non-fraud)")
print("\nSummary plot")
shap.summary_plot(
    shap_values_class_0, 
    X_test, 
    feature_names=feature_names, 
    max_display=10, 
    plot_size=(12, 10)
)

explanation = shap.Explanation(
    values=shap_values_class_0,
    base_values=np.repeat(explainer.expected_value[0], X_test.shape[0]),
    data=X_test.values,
    feature_names=feature_names
)

print("\nBeeswarm plot")
shap.plots.beeswarm(
    explanation, 
    max_display=10, 
    plot_size=(12, 10)
)

"""

""" 
print("\nGenerating summary plot...")
print("\nGenerating summary plot for class 1 (fraud)...")
shap.summary_plot(
    shap_values[1], 
    features=X_test, 
    feature_names=feature_names, 
    max_display=10, 
    plot_size=(12, 10)
)

print("\nGenerating summary plot for class 0 (non-fraud)...")
shap.summary_plot(shap_values[0], features=X_test, feature_names=feature_names, max_display=10, plot_size=(12, 10))

# explanation for the beeswarm
explanation = shap.Explanation(
    values=np.array(shap_values[1]),
    base_values=explainer.expected_value[1].numpy(),
    data=X_test.values,
    feature_names=feature_names
)

print("\nGenerating beeswarm plot...")
print("\nGenerating beeswarm plot for class 1 (fraud)...")
shap.plots.beeswarm(shap_values[1], max_display=10, plot_size=(12, 10))

print("\nGenerating beeswarm plot for class 0 (non-fraud)...")
shap.plots.beeswarm(explanation, max_display=10, plot_size=(12, 10))

 """