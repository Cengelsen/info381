# Imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.advanced import clean_data
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
from sklearn.preprocessing import StandardScaler

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
""" 
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', MLPClassifier(random_state=42))
])

## Define SMOTE parameter grid
param_grid = {
    # SMOTE parameters (unchanged)
    'smote__k_neighbors': [3, 5, 7],
    'smote__sampling_strategy': [0.1, 0.2, 0.3],

    # Common neural network parameters
    'classifier__hidden_layer_sizes': [(32, 16), (64,32)],  # Keras equivalent: Dense layer units
    'classifier__activation': ['relu'],             # Keras: activation functions
    'classifier__learning_rate_init': [0.001, 0.01],        # Keras: learning_rate in optimizer
    'classifier__batch_size': [32, 64],                     # Keras: batch_size
}

## Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1',  # For imbalanced datasets, f1 is often a better metric
    cv=2,
    n_jobs=5,
    verbose=2
)

# Fit the grid search
print("Starting grid search...")
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("\nBest parameters found:")
print(grid_search.best_params_)
print(f"Best score: {grid_search.best_score_:.4f}")
 """

""" {'classifier__activation': 'relu', 'classifier__batch_size': 32, 'classifier__hidden_layer_sizes': (32, 16), 'classifier__learning_rate_init': 0.001, 'smote__k_neighbors': 3, 'smote__sampling_strategy': 0.1} """


smote = SMOTE(
    random_state=42,
)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train) 

# Create and train neural network
nn = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=X_train.shape[1:]),
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
ConfusionMatrixDisplay(cm, display_labels=["Not Fraud", "Fraud"]).plot(cmap="Blues")
plt.show()
plt.savefig("confusion_matrix.png")