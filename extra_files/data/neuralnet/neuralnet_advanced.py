# Imports
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.advanced import clean_data_advanced
from preprocessing.simple import clean_data_simple
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
import shap
from tqdm import tqdm
from sklearn.utils import shuffle

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

# Data loading

print("Importing and cleaning data...")
data = clean_data("fraud.csv")
data = data.sample(frac=1).reset_index(drop=True)

# train test splitting
print("Splitting data...")
X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SMOTE Sampling with Grid Search

## Define the pipeline

smote = SMOTE(
    random_state=42,
)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train) 

# Create and train neural network
nn = tf.keras.models.Sequential([
    tf.keras.Input(shape=X_train.shape[1:]),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

nn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'), 
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc')]
)

# --- 7. Train Model ---
history = nn.fit(
    X_train_smote, y_train_smote,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=2
)


y_pred_prob = nn.predict(X_test).flatten()
threshold = 0.2  # Try different values
y_pred = (y_pred_prob >= threshold).astype(int)
print(classification_report(y_test, y_pred, digits=4))


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Not Fraud", "Fraud"]).plot(cmap="Blues")
disp.plot()
plt.savefig("confusion_matrix.png")
plt.close()


#----------------------------------------------------------------------------------
# CALCULATING SHAPLEY VALUES
#----------------------------------------------------------------------------------

print(type(X_test))

background = X_train_smote[np.random.choice(X_train_smote.shape[0], 1000, replace=False)]

print(f"Using {len(background)} samples for SHAP analysis")

# Initialize SHAP explainer with TensorFlow model
print("Initializing SHAP DeepExplainer...")
explainer = shap.DeepExplainer(nn, background)

# Calculate SHAP values
print("Calculating SHAP values...")
shap_values = explainer.shap_values(X_test)


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

feature_names = X.columns.tolist()

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
                 max_display=10, show=False, plot_type="bar")
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
    data=X_test,
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
    data=X_test,
    feature_names=feature_names
)
shap.plots.beeswarm(explanation_fraud, max_display=10, show=False)
plt.tight_layout()
plt.savefig("shap_beeswarm_fraud.png")
plt.close()

print("All SHAP plots saved to disk.")

#----------------------------------------------------------------------------------
# CALCULATING ANCHOR VALUES
#----------------------------------------------------------------------------------

from alibi.explainers import AnchorTabular

def predict_fn(x):
    # Returns predicted class labels
    return (nn.predict(x, verbose=0) >= 0.5).astype(int).flatten()


print(feature_names)

explainer = AnchorTabular(predict_fn, feature_names)
explainer.fit(X_train_smote)

instance = X_test[100]
explanation = explainer.explain(instance, verbose=True, threshold=0.80)

print("Anchor explanation:")
print(explanation)

print("\n--- Anchor Explanation ---")
print(f"Anchor rule: {' AND '.join(explanation.anchor)}")

X_test_sample = shuffle(X_test, random_state=42)

from collections import Counter

anchors = []
for i in tqdm(range(len(X_test_sample[0:499]))):  # or len(X_test)
    instance = X_test_sample[i]
    explanation = explainer.explain(
        instance, 
        threshold=0.80, 
        verbose=False,
        tau=0.1,
        batch_size=1024,
        )
    anchors.append(tuple(explanation.anchor))  # store as tuple for counting

anchor_counts = Counter(anchors)
print("Most common anchor rules:")
for rule, count in anchor_counts.most_common(10):
    print(f"{rule}: {count} times")