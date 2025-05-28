import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np
import os
from common import * # Assuming common.py contains data_loader, apply_smote, apply_pca

# Paths & Environment Variables
base_path = os.getenv("base_path", '../../Data/training')
training_model = os.getenv('training_model', 'BoW')
data_type = os.getenv('data_type', 'ZS')
exp = os.getenv("exp", "folder")
FS = os.getenv("feature_selection", 'no')=='yes'
num_features = int(os.getenv("num_features", "15"))
smote_ing = os.getenv('smote_ing', 'no')
pca_ing   = os.getenv('pca_ing', 'no')

# Load data
X_train, Y_train, X_test, Y_test = data_loader(base_path, training_model, data_type, feature_selection = FS, num_features=num_features)

print(f'Before SMOTE: {X_train.shape, len(Y_train)}, {X_test.shape, len(Y_test)}')

# Apply SMOTE (if needed)
if smote_ing == "yes":
    from imblearn.over_sampling import SMOTE
    smote = SMOTE()
    X_train, Y_train = smote.fit_resample(X_train, Y_train)

    print(f'After SMOTE: {X_train.shape, len(Y_train)}, {X_test.shape, len(Y_test)}')

# Apply PCA (if needed)
if pca_ing == "yes":
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print(f'After PCA: {X_train.shape}, {X_test.shape}')

# Train XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    n_jobs=-1
)

xgb_model.fit(X_train, Y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Evaluate model
accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred, average='binary', zero_division=0)
recall = recall_score(Y_test, y_pred, average='binary', zero_division=0)
f1 = f1_score(Y_test, y_pred, average='binary')
auc = roc_auc_score(Y_test, y_pred_proba)

cm = confusion_matrix(Y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")
print(f"Classification Report:\n{classification_report(Y_test, y_pred)}")
print(f"DataType, Accuracy, Precision, Recall, F1 Score, AUC")
print(f"{data_type}, {accuracy:.4f}, {precision:.4f}, {recall:.4f}, {f1:.4f}, {auc:.4f}")