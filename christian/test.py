import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.advanced import clean_data
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasClassifier

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

# Apply SMOTE to the training data only
print("Applying SMOTE oversampling...")
smote = SMOTE(
    random_state=42, 
    sampling_strategy=0.5,
    k_neighbors=3,
)


""" 
Best parameters:
  k_neighbors: 3.0
  sampling_strategy: 0.5
  Achieved F1: 0.0158
  Precision: 0.0136
  Recall: 0.0186
  Optimal threshold: 0.3342
"""
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Training set shape after SMOTE: {X_train_smote.shape}")
print(f"Training set class distribution after SMOTE: {Counter(y_train_smote)}")


# Define the model-building function for SciKeras
def create_model(meta, hidden_layer_sizes=[64, 32], dropout_rate=0.2, 
                 activation='relu', learning_rate=0.001):
    """Create a Keras model with the given parameters."""
    # Get input shape from metadata
    n_features_in_ = meta["n_features_in_"]
    
    model = keras.Sequential()
    
    # Input layer
    model.add(keras.layers.Dense(hidden_layer_sizes[0], 
                                 input_shape=(n_features_in_,), 
                                 activation=activation))
    model.add(keras.layers.Dropout(dropout_rate))
    
    # Hidden layers
    for units in hidden_layer_sizes[1:]:
        model.add(keras.layers.Dense(units, activation=activation))
        model.add(keras.layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    # Compile the model
    optimizer = keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', 
                  optimizer=optimizer, 
                  metrics=['accuracy'])
    
    return model

# Create a SciKeras wrapper for our model
# Note: In newer versions, we need to be more explicit
model = KerasClassifier(
    model=create_model,
    hidden_layer_sizes=[64, 32],
    dropout_rate=0.2,
    activation='relu',
    learning_rate=0.001,
    verbose=0,
    random_state=42
)

# Define parameter grid
param_grid = {
    'batch_size': [16],  # Reduced for quicker execution
    'epochs': [10],          # Reduced for quicker execution
    'model__hidden_layer_sizes': [[32, 16]],
    'model__dropout_rate': [0.2],
    'model__activation': ['relu'],
    'model__learning_rate': [0.001]
}

# Create cross-validation strategy explicitly
cv = KFold(n_splits=2, shuffle=True, random_state=42)

# Create GridSearchCV with explicit cv parameter
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=4,
    verbose=2
)

""" Rank 1: Score=0.6667, Params={'batch_size': 16, 'epochs': 10, 'model__activation': 'relu', 'model__dropout_rate': 0.2, 'model__hidden_layer_sizes': [32, 16], 'model__learning_rate': 0.001} """

# Fit the grid search
print("Starting grid search...")
try:
    grid_result = grid.fit(X_train_smote, y_train_smote)

    # Print results
    print(f"\nBest score: {grid_result.best_score_:.4f}")
    print(f"Best parameters: {grid_result.best_params_}")

    # Evaluate the best model on the test set
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    print("\nTest Set Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Print top 5 parameter combinations
    print("\nTop 5 parameter combinations:")
    results_df = pd.DataFrame(grid_result.cv_results_)
    top_results = results_df.sort_values(by='rank_test_score').head(5)
    for i, row in top_results.iterrows():
        print(f"Rank {row['rank_test_score']}: Score={row['mean_test_score']:.4f}, Params={row['params']}")
        
except Exception as e:
    print(f"Error occurred: {e}")
    print("\nFalling back to manual hyperparameter tuning...")
    
    # If grid search fails, demonstrate a manual hyperparameter search
    best_score = 0
    best_params = {}
    
    # Simplified parameter combinations
    param_combinations = [
        {'hidden_layer_sizes': [64, 32], 'dropout_rate': 0.2, 'activation': 'relu', 'learning_rate': 0.001},
        {'hidden_layer_sizes': [32, 16], 'dropout_rate': 0.3, 'activation': 'tanh', 'learning_rate': 0.01}
    ]
    
    for params in param_combinations:
        print(f"\nTrying parameters: {params}")
        # Create and train model with these parameters
        model = KerasClassifier(
            model=create_model,
            hidden_layer_sizes=params['hidden_layer_sizes'],
            dropout_rate=params['dropout_rate'],
            activation=params['activation'],
            learning_rate=params['learning_rate'],
            epochs=10,
            batch_size=32,
            verbose=0,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test accuracy: {accuracy:.4f}")
        
        if accuracy > best_score:
            best_score = accuracy
            best_params = params
    
    print(f"\nBest manual parameters: {best_params}")
    print(f"Best manual score: {best_score:.4f}")