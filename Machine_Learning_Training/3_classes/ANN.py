import numpy as np
import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from common import * 

# Simple ANN Model
class ANN_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ANN_Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Prepare the data for ANN

def prepare_data(X_train, X_test, Y_train, Y_test, batch_size=64):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    train_data = TensorDataset(X_train_tensor, Y_train_tensor)
    test_data = TensorDataset(X_test_tensor, Y_test_tensor)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, Y_test_tensor

# Train the model and save the best one

def train_model(train_loader, test_loader, model, criterion, optimizer, num_epochs, device, data_type):
    best_f1 = 0.0
    best_model_state_dict = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = running_loss / len(train_loader)
        f1, accuracy, precision, recall, auc = evaluate_model(test_loader, model, device, return_all_metrics=True)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_state_dict = model.state_dict().copy()  # Save the best model state dict
            print(f"Best model with F1 Score: {best_f1:.4f}")
    
    # Return the best model's state_dict
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
    else:
        print("No best model found based on F1 score.")
    return model


def evaluate_model(test_loader, model, device, return_all_metrics=False):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    if return_all_metrics:
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        auc = roc_auc_score(all_labels, all_probs, average='weighted', multi_class='ovr')
        return f1, accuracy, precision, recall, auc
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    auc = roc_auc_score(all_labels, all_probs, average='weighted', multi_class='ovr')                        
    
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{classification_report(all_labels, all_preds)}")
    print(f"DataType, Accuracy, Precision, Recall, F1 Score, AUC")
    print(f"ANN, {data_type}, {accuracy:.4f}, {precision:.4f}, {recall:.4f}, {f1:.4f}, {auc:.4f}")


# Paths & Environment Variables
base_path = os.getenv("base_path", '../../Data/training')
training_model = os.getenv('training_model', 'BoW')
data_type = os.getenv('data_type', 'ZS')
exp = os.getenv("exp", "folder")
FS = os.getenv("feature_selection", 'no')=='yes'
num_features = int(os.getenv("num_features", "15"))
k_bow = int(os.getenv("k_bow", "8"))

smote_ing = os.getenv('smote_ing', 'no')
pca_ing   = os.getenv('pca_ing', 'no')

print(f"Base Path: {base_path}, Training Model: {training_model}, Data Type: {data_type}, Experiment: {exp}")
print(f"Feature Selection: {FS}, Number of Features: {num_features}, K BoW: {k_bow}")
print(f"SMOTE: {smote_ing}, PCA: {pca_ing}")

num_epochs = int(os.getenv("epochs", 25))
learning_rate = float(os.getenv("learning_rate", 0.001))

print(f'Dealing with {num_features} features...')
print(f'data type = {data_type}\nSMOTE = {smote_ing}\nPCA = {pca_ing}\nNumber of epochs = {num_epochs}\nLearning Rate = {learning_rate}')

X_train, Y_train, X_test, Y_test = data_loader(base_path, training_model, data_type, feature_selection = FS, num_features=num_features, k_bow=k_bow)
print(f'Before SMOTE: {X_train.shape, len(Y_train)}, {X_test.shape, len(Y_test)}')

if smote_ing == "yes":
    from imblearn.over_sampling import SMOTE
    smote = SMOTE()
    X_train, Y_train = smote.fit_resample(X_train, Y_train)
    print(f'After SMOTE: {X_train.shape, len(Y_train)}, {X_test.shape, len(Y_test)}')

if pca_ing == "yes":
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print(f'After PCA: {X_train.shape}, {X_test.shape}')

train_loader, test_loader, Y_test_tensor = prepare_data(X_train, X_test, Y_train, Y_test)

input_size = X_train.shape[1]
hidden_size = 64
num_classes = len(set(Y_train))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = ANN_Model(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

saving_path = "./model_checkpoints"
os.makedirs(saving_path, exist_ok=True)

best_model = train_model(train_loader, test_loader, model, criterion, optimizer, num_epochs, device, data_type)

evaluate_model(test_loader, best_model, device)