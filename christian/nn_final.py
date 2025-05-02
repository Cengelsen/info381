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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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


# SMOTE Sampling with Grid Search

## Define the pipeline

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


""" {'classifier__activation': 'relu', 'classifier__batch_size': 32, 'classifier__hidden_layer_sizes': (32, 16), 'classifier__learning_rate_init': 0.001, 'smote__k_neighbors': 3, 'smote__sampling_strategy': 0.1} """

# Apply SMOTE to the training data only
print("Applying SMOTE oversampling...")
smote = SMOTE(
    random_state=42, 
    sampling_strategy=grid_search.best_params_['smote__sampling_strategy'],
    k_neighbors=grid_search.best_params_['smote__k_neighbors'],
)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Create and train neural network
nn = tf.keras.models.Sequential([
    tf.keras.layers.Dense(grid_search.best_params_['classifier__hidden_layer_sizes'][0][0], activation='relu', input_shape=X_train.shape[1:]),
    tf.keras.layers.Dense(grid_search.best_params_['classifier__hidden_layer_sizes'][0][1], activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

nn.compile(optimizer='adam',
           loss='binary_crossentropy',
           metrics=['accuracy'])

nn.fit(X_train_smote, y_train_smote, epochs=10, batch_size=32)

# Get probability predictions
y_pred_proba = nn.predict(X_test)

# Convert probabilities to binary predictions (0 or 1) using threshold of 0.5
y_pred_binary = (y_pred_proba > 0.5).astype(int)

plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test, y_pred_binary)
ConfusionMatrixDisplay(cm, display_labels=["Not Fraud", "Fraud"]).plot(cmap="Blues")
plt.title('Confusion Matrix with SMOTE')
plt.savefig("confusion_matrix_smote.png")
plt.close()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))


nn.fit(X_train, y_train, epochs=10, batch_size=32)

# Get probability predictions
y_pred_proba = nn.predict(X_test)

# Convert probabilities to binary predictions (0 or 1) using threshold of 0.5
y_pred_binary = (y_pred_proba > 0.5).astype(int)

plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test, y_pred_binary)
ConfusionMatrixDisplay(cm, display_labels=["Not Fraud", "Fraud"]).plot(cmap="Blues")
plt.title('Confusion Matrix without SMOTE')
plt.savefig("confusion_matrix_no_smote.png")
plt.close()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))

# Training final model with best hyperparameters

# Evaluating final model on test set

# Saving final model
