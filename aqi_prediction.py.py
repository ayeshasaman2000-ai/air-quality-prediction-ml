# =============================================================================
# Air Quality Prediction Using Machine Learning
# Author: Ayesha Saman | MS Computer Science, BZU Multan
# Models: Neural Network, XGBoost, Random Forest, SVM, KNN
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import xgboost as xgb
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, f1_score,
    precision_score, recall_score
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# =============================================================================
# 1. LOAD DATA
# =============================================================================

# Place city_day.csv in the same folder as this script before running
data = pd.read_csv("city_day.csv")
print("Dataset loaded successfully.")
print(f"Shape: {data.shape}\n")

# =============================================================================
# 2. FEATURE & TARGET COLUMNS
# =============================================================================

FEATURE_COLS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
                'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
TARGET_COL   = 'AQI_Bucket'
ALL_COLS     = FEATURE_COLS + [TARGET_COL]

# Keep only relevant columns and drop rows missing the target label
data = data[ALL_COLS].dropna(subset=[TARGET_COL])

# Impute missing feature values with column mean
imputer = SimpleImputer(strategy='mean')
data[FEATURE_COLS] = imputer.fit_transform(data[FEATURE_COLS])

print(f"Rows after cleaning: {len(data)}")
print(f"Class distribution:\n{data[TARGET_COL].value_counts()}\n")

# =============================================================================
# 3. PREPARE FEATURES AND LABELS
# =============================================================================

X = data[FEATURE_COLS].values

# --- For Neural Network: one-hot encode labels ---
y_onehot = pd.get_dummies(data[TARGET_COL]).values
class_names_onehot = pd.get_dummies(data[TARGET_COL]).columns.tolist()

# --- For classical models: integer encode labels ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data[TARGET_COL])

# =============================================================================
# 4. TRAIN / TEST SPLIT & SCALING
# =============================================================================

scaler = MinMaxScaler()

# Split for Neural Network (one-hot labels)
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y_onehot
)
X_train_nn = scaler.fit_transform(X_train_nn)
X_test_nn  = scaler.transform(X_test_nn)   # use transform (not fit_transform) on test set

# Split for classical models (integer labels)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)         # use transform (not fit_transform) on test set

# =============================================================================
# 5. NEURAL NETWORK
# =============================================================================

print("=" * 50)
print("Training Neural Network...")
print("=" * 50)

nn_model = Sequential([
    Dense(256, input_dim=X_train_nn.shape[1], activation='relu'),
    Dense(128, activation='relu'),
    Dense(64,  activation='relu'),
    Dense(32,  activation='relu'),
    Dense(16,  activation='relu'),
    Dense(16,  activation='relu'),
    Dense(8,   activation='relu'),
    Dense(8,   activation='relu'),
    Dense(len(class_names_onehot), activation='softmax'),
])

optimizer = Adam(learning_rate=0.015)
nn_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = nn_model.fit(
    X_train_nn, y_train_nn,
    validation_split=0.2,
    epochs=30,
    batch_size=128,
    verbose=1
)

nn_loss, nn_accuracy = nn_model.evaluate(X_test_nn, y_test_nn, verbose=0)
print(f"\nNeural Network Test Accuracy: {nn_accuracy * 100:.2f}%\n")

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'],     label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Neural Network — Training vs Validation Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('nn_training_history.png', dpi=150)
plt.show()

# Confusion matrix for Neural Network
y_pred_nn    = nn_model.predict(X_test_nn)
y_pred_nn    = np.argmax(y_pred_nn, axis=1)
y_true_nn    = np.argmax(y_test_nn,  axis=1)
cls_indices  = list(range(len(class_names_onehot)))

matrix_nn = confusion_matrix(y_true_nn, y_pred_nn, labels=cls_indices)

plt.figure(figsize=(8, 6))
sns.heatmap(matrix_nn, fmt='g', cmap='Greens', annot=True,
            xticklabels=class_names_onehot, yticklabels=class_names_onehot)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Neural Network — Confusion Matrix')
plt.tight_layout()
plt.savefig('nn_confusion_matrix.png', dpi=150)
plt.show()

print("Neural Network Classification Report:")
print(classification_report(y_true_nn, y_pred_nn, target_names=class_names_onehot))

# =============================================================================
# 6. XGBOOST
# =============================================================================

print("=" * 50)
print("Training XGBoost...")
print("=" * 50)

xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(label_encoder.classes_),
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_pred) * 100:.2f}%")
print(classification_report(y_test, xgb_pred, target_names=label_encoder.classes_))

# =============================================================================
# 7. KNN CLASSIFIER
# =============================================================================

print("=" * 50)
print("Training KNN...")
print("=" * 50)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

print(f"KNN Accuracy: {accuracy_score(y_test, knn_pred) * 100:.2f}%")
print(classification_report(y_test, knn_pred, target_names=label_encoder.classes_))

# =============================================================================
# 8. SVM CLASSIFIER
# =============================================================================

print("=" * 50)
print("Training SVM...")
print("=" * 50)

svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

print(f"SVM Accuracy: {accuracy_score(y_test, svm_pred) * 100:.2f}%")
print(classification_report(y_test, svm_pred, target_names=label_encoder.classes_))

# =============================================================================
# 9. RANDOM FOREST
# =============================================================================

print("=" * 50)
print("Training Random Forest...")
print("=" * 50)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred) * 100:.2f}%")
print(classification_report(y_test, rf_pred, target_names=label_encoder.classes_))

# =============================================================================
# 10. MODEL COMPARISON SUMMARY
# =============================================================================

print("\n" + "=" * 50)
print("MODEL COMPARISON SUMMARY")
print("=" * 50)

results = {
    'Neural Network': nn_accuracy * 100,
    'XGBoost':        accuracy_score(y_test, xgb_pred) * 100,
    'KNN':            accuracy_score(y_test, knn_pred) * 100,
    'SVM':            accuracy_score(y_test, svm_pred) * 100,
    'Random Forest':  accuracy_score(y_test, rf_pred)  * 100,
}

for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {model_name:<20} {acc:.2f}%")

# Bar chart comparison
plt.figure(figsize=(9, 5))
plt.bar(results.keys(), results.values(), color='steelblue', edgecolor='white')
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison — AQI Prediction')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

print("\nDone. Charts saved as PNG files in the current folder.")
