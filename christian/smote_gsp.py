from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.advanced import clean_data

# Importing data
print("Importing and cleaning data...")
data = clean_data("fraud.csv")
data = data.sample(frac=1).reset_index(drop=True)

# Check original class distribution
print("Original class distribution:")
print(data["is_fraud"].value_counts())
print(f"Fraud percentage: {data['is_fraud'].mean()*100:.2f}%")

# Splitting data - IMPORTANT: Split first, then apply SMOTE only to training data
# We keep the test set with original distribution to evaluate real-world performance
X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

# Use stratify=y to ensure both splits have the same class proportion
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Training set shape before SMOTE: {X_train.shape}")
print(f"Training set class distribution before SMOTE: {Counter(y_train)}")

# ------------------- SMOTE GRID SEARCH -------------------

# Define a function to create and train neural network
def create_and_train_nn(X_train_smote, y_train_smote, X_val, y_val):
    nn = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_smote.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    nn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    # Add early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Track the learning history
    nn.fit(
        X_train_smote, y_train_smote,
        epochs=10,  # Reduced for grid search
        batch_size=512,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0  # Reduce output during grid search
    )
    
    return nn

# Split training data into training and validation sets for grid search
X_train_gs, X_val, y_train_gs, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print("\n---------- Starting SMOTE Grid Search ----------")

# Define SMOTE parameter grid
smote_params = {
    'k_neighbors': [3, 5, 7, 10],
    'sampling_strategy': [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],  # Ratio of minority to majority class
}

# Initialize results storage
results = []

# For each parameter combination, run SMOTE and evaluate model
total_combinations = len(smote_params['k_neighbors']) * len(smote_params['sampling_strategy'])
current_combo = 0

for k in smote_params['k_neighbors']:
    for ratio in smote_params['sampling_strategy']:
        current_combo += 1
        print(f"\nTesting combination {current_combo}/{total_combinations}: k_neighbors={k}, sampling_strategy={ratio}")
        
        # Configure SMOTE with current parameters
        smote = SMOTE(random_state=42, k_neighbors=k, sampling_strategy=ratio)
        
        # Apply SMOTE
        X_train_smote, y_train_smote = smote.fit_resample(X_train_gs, y_train_gs)
        
        print(f"  Resampled class distribution: {Counter(y_train_smote)}")
        
        # Train model
        model = create_and_train_nn(X_train_smote, y_train_smote, X_val, y_val)
        
        # Evaluate on validation set
        y_pred_proba = model.predict(X_val)
        
        # Find optimal threshold using precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        # Apply optimal threshold
        y_pred = (y_pred_proba >= optimal_threshold).astype(int).flatten()
        
        # Calculate metrics
        val_f1 = f1_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Store results
        results.append({
            'k_neighbors': k,
            'sampling_strategy': ratio,
            'f1_score': val_f1,
            'precision': precision,
            'recall': recall,
            'optimal_threshold': optimal_threshold,
            'resampled_ratio': Counter(y_train_smote)[1] / Counter(y_train_smote)[0]
        })
        
        print(f"  Validation F1: {val_f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# Find best parameters
results_df = pd.DataFrame(results)
best_result = results_df.loc[results_df['f1_score'].idxmax()]

print("\n---------- SMOTE Grid Search Results ----------")
print(f"Best parameters:")
print(f"  k_neighbors: {best_result['k_neighbors']}")
print(f"  sampling_strategy: {best_result['sampling_strategy']}")
print(f"  Achieved F1: {best_result['f1_score']:.4f}")
print(f"  Precision: {best_result['precision']:.4f}")
print(f"  Recall: {best_result['recall']:.4f}")
print(f"  Optimal threshold: {best_result['optimal_threshold']:.4f}")

# Plot grid search results
plt.figure(figsize=(15, 10))

# Plot F1 scores
plt.subplot(2, 2, 1)
for k in smote_params['k_neighbors']:
    k_results = results_df[results_df['k_neighbors'] == k]
    plt.plot(k_results['sampling_strategy'], k_results['f1_score'], 'o-', label=f'k={k}')
plt.xlabel('Sampling Strategy')
plt.ylabel('F1 Score')
plt.title('F1 Score by SMOTE Parameters')
plt.legend()
plt.grid(True)

# Plot Precision
plt.subplot(2, 2, 2)
for k in smote_params['k_neighbors']:
    k_results = results_df[results_df['k_neighbors'] == k]
    plt.plot(k_results['sampling_strategy'], k_results['precision'], 'o-', label=f'k={k}')
plt.xlabel('Sampling Strategy')
plt.ylabel('Precision')
plt.title('Precision by SMOTE Parameters')
plt.legend()
plt.grid(True)

# Plot Recall
plt.subplot(2, 2, 3)
for k in smote_params['k_neighbors']:
    k_results = results_df[results_df['k_neighbors'] == k]
    plt.plot(k_results['sampling_strategy'], k_results['recall'], 'o-', label=f'k={k}')
plt.xlabel('Sampling Strategy')
plt.ylabel('Recall')
plt.title('Recall by SMOTE Parameters')
plt.legend()
plt.grid(True)

# Plot Optimal Thresholds
plt.subplot(2, 2, 4)
for k in smote_params['k_neighbors']:
    k_results = results_df[results_df['k_neighbors'] == k]
    plt.plot(k_results['sampling_strategy'], k_results['optimal_threshold'], 'o-', label=f'k={k}')
plt.xlabel('Sampling Strategy')
plt.ylabel('Optimal Threshold')
plt.title('Optimal Threshold by SMOTE Parameters')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('smote_grid_search_results.png')
plt.close()

# ------------------- Train final model with best SMOTE parameters -------------------
print("\n---------- Training Final Model with Best SMOTE Parameters ----------")

# Configure SMOTE with best parameters
best_smote = SMOTE(
    random_state=42, 
    k_neighbors=int(best_result['k_neighbors']),
    sampling_strategy=float(best_result['sampling_strategy'])
)

# Apply SMOTE with best parameters to full training set
X_train_best_smote, y_train_best_smote = best_smote.fit_resample(X_train, y_train)

print(f"Training set shape after SMOTE: {X_train_best_smote.shape}")
print(f"Training set class distribution after SMOTE: {Counter(y_train_best_smote)}")

# Visualize class distribution before and after SMOTE
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(['Not Fraud', 'Fraud'], [Counter(y_train)[0], Counter(y_train)[1]])
plt.title('Class Distribution Before SMOTE')
plt.ylabel('Number of Samples')

plt.subplot(1, 2, 2)
plt.bar(['Not Fraud', 'Fraud'], [Counter(y_train_best_smote)[0], Counter(y_train_best_smote)[1]])
plt.title(f'Class Distribution After SMOTE\n(k={int(best_result["k_neighbors"])}, ratio={float(best_result["sampling_strategy"]):.2f})')

plt.tight_layout()
plt.savefig('best_smote_distribution.png')
plt.close()

# Create and train neural network with optimal SMOTE-balanced data
print("Training neural network with optimized SMOTE-balanced data...")
nn = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_best_smote.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

nn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Add early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Track the learning history
history = nn.fit(
    X_train_best_smote, y_train_best_smote,
    epochs=15,
    batch_size=512,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Plot training history
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics_best_smote.png')
plt.close()

# Evaluate on the test set (which has the original distribution)
print("Evaluating on the test set (with original distribution)...")
y_pred_proba = nn.predict(X_test)

# Find optimal threshold using precision-recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Plot precision-recall curve
plt.figure(figsize=(10, 6))
plt.plot(recalls, precisions, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Optimal SMOTE)')
plt.grid(True)
plt.savefig("precision_recall_curve_best_smote.png")
plt.close()

# Find threshold that maximizes F1 score
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
print(f"Optimal threshold: {optimal_threshold:.4f}")

# Apply optimal threshold for predictions
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int).flatten()

# Evaluate with optimal threshold
print("\nResults with optimal threshold:")
cm_optimal = confusion_matrix(y_test, y_pred_optimal)

plt.figure(figsize=(10, 7))
ConfusionMatrixDisplay(cm_optimal, display_labels=["Not Fraud", "Fraud"]).plot(cmap="Blues")
plt.title(f'Confusion Matrix (threshold={optimal_threshold:.4f})')
plt.savefig("confusion_matrix_best_smote.png")
plt.close()

print("\nClassification Report (optimal threshold):")
print(classification_report(y_test, y_pred_optimal))

# Also show results with standard threshold for comparison
y_pred_standard = (y_pred_proba >= 0.5).astype(int).flatten()
print("\nClassification Report (standard threshold=0.5):")
print(classification_report(y_test, y_pred_standard))

# Save model if needed
nn.save('fraud_detection_model_best_smote.h5')

# Compare with original SMOTE approach (optional)
print("\n---------- Comparison: Original vs Optimized SMOTE ----------")
print(f"Best SMOTE Parameters: k_neighbors={int(best_result['k_neighbors'])}, sampling_strategy={float(best_result['sampling_strategy']):.2f}")
print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"Test set F1 score: {f1_score(y_test, y_pred_optimal):.4f}")