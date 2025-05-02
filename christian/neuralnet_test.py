from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import sys
from imblearn.over_sampling import SMOTE
from collections import Counter
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

# Visualize class distribution before and after SMOTE
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(['Not Fraud', 'Fraud'], [Counter(y_train)[0], Counter(y_train)[1]])
plt.title('Class Distribution Before SMOTE')
plt.ylabel('Number of Samples')

plt.subplot(1, 2, 2)
plt.bar(['Not Fraud', 'Fraud'], [Counter(y_train_smote)[0], Counter(y_train_smote)[1]])
plt.title('Class Distribution After SMOTE')

plt.tight_layout()
plt.savefig('smote_distribution.png')
plt.close()

# Create and train neural network with balanced data
print("Training neural network with SMOTE-balanced data...")
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
history = nn.fit(
    X_train_smote, y_train_smote,
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

plt.subplot(1, 3, 3)
plt.plot(history.history['precision'], label='Training Precision')
plt.plot(history.history['recall'], label='Training Recall')
plt.plot(history.history['val_precision'], label='Val Precision')
plt.plot(history.history['val_recall'], label='Val Recall')
plt.title('Precision & Recall')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics_smote.png')
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
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.savefig("precision_recall_curve_smote.png")
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
plt.savefig("confusion_matrix_smote.png")
plt.close()

print("\nClassification Report (optimal threshold):")
print(classification_report(y_test, y_pred_optimal))

# Also show results with standard threshold for comparison
y_pred_standard = (y_pred_proba >= 0.5).astype(int).flatten()
print("\nClassification Report (standard threshold=0.5):")
print(classification_report(y_test, y_pred_standard))

# Calculate and print key metrics at different thresholds
print("\nMetrics at different thresholds:")
thresholds_to_try = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, optimal_threshold]
results = []

for threshold in thresholds_to_try:
    y_pred = (y_pred_proba >= threshold).astype(int).flatten()
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    result = {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }
    results.append(result)
    
    print(f"Threshold: {threshold:.2f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, TP: {tp}, FP: {fp}, FN: {fn}")

# Save model if needed
# nn.save('fraud_detection_model_smote.h5')

# Optional: Plot metrics vs thresholds
plt.figure(figsize=(10, 6))
thresholds = [r['threshold'] for r in results]
precisions = [r['precision'] for r in results]
recalls = [r['recall'] for r in results]
f1s = [r['f1'] for r in results]

plt.plot(thresholds, precisions, 'o-', label='Precision')
plt.plot(thresholds, recalls, 'o-', label='Recall')
plt.plot(thresholds, f1s, 'o-', label='F1 Score')
plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.2f})')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Metrics vs. Threshold')
plt.legend()
plt.grid(True)
plt.savefig('metrics_vs_threshold.png')
plt.close()