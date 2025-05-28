import pandas as pd
import numpy as np
import os
import pickle
import torch
from imblearn.over_sampling import SMOTE

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from collections import Counter

# =========================================================
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.preprocessing import MinMaxScaler # Needed for chi2 demo if data isn't guaranteed non-negative
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# =========================================================

def feature_selection_SelectKBest(X_train, y_train, X_test, k_best=15):
    # --- 3. Example: SelectKBest ---
    # Select the top 15 features using ANOVA F-value (suitable for numerical)
    # Note: Applying f_classif to counts often works okay, but chi2/MI might be theoretically better for BOW
    print(f"\n--- Using SelectKBest (k={k_best}, score_func=f_classif) ---")

    selector_kbest = SelectKBest(score_func=f_classif, k=k_best)

    # Fit only on the training data
    selector_kbest.fit(X_train, y_train)

    # # Get the selected feature indices and names
    # selected_indices_kbest = selector_kbest.get_support(indices=True)
    # selected_features_kbest = X_train.columns[selected_indices_kbest]
    # print(f"Selected feature names ({len(selected_features_kbest)}): {selected_features_kbest.tolist()}")

    # Transform both train and test sets
    X_train_kbest = selector_kbest.transform(X_train)
    X_test_kbest = selector_kbest.transform(X_test)

    print(f"Shape after SelectKBest: Train={X_train_kbest.shape}, Test={X_test_kbest.shape}")
    print("-" * 30)
    return X_train_kbest, X_test_kbest

# =========================================================
def feature_selection_SelectPercentile(X_train, y_train, X_test, percentile_best = 20):
    # --- 4. Example: SelectPercentile ---
    # Select the top 20% of features using mutual information

    print(f"\n--- Using SelectPercentile (percentile={percentile_best}, score_func=mutual_info_classif) ---")

    selector_percentile = SelectPercentile(score_func=mutual_info_classif, percentile=percentile_best)

    # Fit only on the training data
    # Mutual information can take a bit longer
    selector_percentile.fit(X_train, y_train)

    # Get the selected feature indices and names
    selected_indices_percentile = selector_percentile.get_support(indices=True)
    selected_features_percentile = X_train.columns[selected_indices_percentile]
    print(f"Selected feature names ({len(selected_features_percentile)}): {selected_features_percentile.tolist()}")

    # Transform both train and test sets
    X_train_percentile = selector_percentile.transform(X_train)
    X_test_percentile = selector_percentile.transform(X_test)

    print(f"Shape after SelectPercentile: Train={X_train_percentile.shape}, Test={X_test_percentile.shape}")
    print("-" * 30)
    return X_train_precentile, X_test_percentile


# =========================================================
def feature_selection_SelectKBest_CHI2_BoW(X_train, y_train, X_test,k_bow = 200):
    # --- 5. Example: Using chi2 (Appropriate for BOW Features) ---
    # Let's specifically select from the "BOW-like" features using chi2
    print("\n--- Using SelectKBest with chi2 (on BOW features only) ---")

    # # Identify BOW feature columns
    # bow_columns = [col for col in X_train.columns if col.startswith('bow_')]
    # X_train_bow_only = X_train[bow_columns]
    # X_test_bow_only = X_test[bow_columns]

    # **Important**: chi2 requires non-negative features. Our simulated BOW are already ints >= 0.
    # If your features might be negative (e.g., TF-IDF centered), scale them first:
    # scaler = MinMaxScaler()
    # X_train_bow_scaled = scaler.fit_transform(X_train_bow_only)
    # X_test_bow_scaled = scaler.transform(X_test_bow_only)
    # For this example, we assume they are already non-negative.
    # X_train_bow_scaled = X_train_bow_only # Assuming non-negative
    # X_test_bow_scaled = X_test_bow_only   # Assuming non-negative

    X_train_bow_scaled = X_train # Assuming non-negative
    X_test_bow_scaled = X_test   # Assuming non-negative

    # Select top 8 BOW features
    selector_chi2 = SelectKBest(score_func=chi2, k=k_bow)

    selector_chi2.fit(X_train_bow_scaled, y_train)

    # selected_indices_chi2 = selector_chi2.get_support(indices=True)
    # selected_features_chi2 = X_train_bow_only.columns[selected_indices_chi2]
    # print(f"Selected BOW feature names ({len(selected_features_chi2)}): {selected_features_chi2.tolist()}")

    X_train_chi2 = selector_chi2.transform(X_train_bow_scaled)
    X_test_chi2 = selector_chi2.transform(X_test_bow_scaled)

    print(f"Shape after SelectKBest (chi2) on BOW: Train={X_train_chi2.shape}, Test={X_test_chi2.shape}")

    # If you wanted to combine these selected BOW features with selected tabular features,
    # you would perform selection on tabular features separately (e.g., using f_classif)
    # and then horizontally stack (np.hstack or pd.concat) the results.
    print("-" * 30)
    return X_train_chi2, X_test_chi2



def apply_pca(X_train, X_test, n_components=None):
    """
    Apply PCA-based feature selection on training data and transform test data accordingly.
    
    Parameters:
    X_train (numpy.ndarray or pandas.DataFrame): Training feature set.
    X_test (numpy.ndarray or pandas.DataFrame): Test feature set.
    n_components (int, float, or None): Number of principal components to keep. 
        - If None, keeps all components.
        - If float (e.g., 0.95), selects the number of components to retain that variance.
        - If int, selects that many principal components.
    
    Returns:
    X_train_pca (numpy.ndarray): Transformed training data.
    X_test_pca (numpy.ndarray): Transformed test data.
    pca (PCA object): The fitted PCA model.
    """
    print('==============================================')
    print("               Applying PCA")
    print('==============================================')
    
    # Standardize the data (PCA works best with normalized data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA on X_train and transform X_test using the same PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, X_test_pca, pca


def apply_smote(X, Y, random_state=42):
    """
    Apply SMOTE to balance classes in a dataset.
    
    Parameters:
    - X: array-like of shape (n_samples, n_features), the input features
    - Y: array-like of shape (n_samples,), the target labels (0, 1, 2)
    - random_state: int, controls reproducibility (default=42)
    
    Returns:
    - X_resampled: array-like, resampled features
    - Y_resampled: array-like, resampled labels
    """
    print('==============================================')
    print("               Applying SMOTE")
    print('==============================================')
    print("Class distribution before SMOTE:", Counter(Y))

    # Ensure X and Y are numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    # Check input validity
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if not set(np.unique(Y)).issubset({0, 1, 2}):
        raise ValueError("Y must contain only 0, 1, or 2")
    
    # Initialize SMOTE
    smote = SMOTE(random_state=random_state)
    
    # Apply SMOTE to resample the dataset
    X_resampled, Y_resampled = smote.fit_resample(X, Y)
    
    print("Class distribution after SMOTE:", Counter(Y_resampled))

    return X_resampled, Y_resampled


def data_loader(base_path, training_model, data_type, feature_selection = False, num_features=15):
    from datasets import load_from_disk
    saving_path = f'{base_path}/{training_model}'
    
    train_dataset = load_from_disk(f'{saving_path}/train_dataset_X')
    test_dataset = load_from_disk(f'{saving_path}/test_dataset_X')
    print(train_dataset)
    
    train_Y = train_dataset['label']
    test_Y  =  test_dataset['label']
    
   
    if data_type =="Z":
        train_X = np.concatenate((train_dataset['XB'], 
                                  train_dataset['XD'], 
                                  train_dataset['XL'], 
                                  train_dataset['XM'], 
                                  train_dataset['XP']), axis=1)
        
        test_X = np.concatenate ((test_dataset['XB'], 
                                  test_dataset['XD'], 
                                  test_dataset['XL'], 
                                  test_dataset['XM'], 
                                  test_dataset['XP']), axis=1)
        
        # Apply SelectKBest
        if feature_selection:
            train_X, test_X = feature_selection_SelectKBest(train_X, train_Y, test_X, k_best=num_features)
        
    elif data_type =="ZS":
        train_X = np.concatenate((train_dataset['XB'], 
                                  train_dataset['XD'], 
                                  train_dataset['XL'], 
                                  train_dataset['XM'], 
                                  train_dataset['XP'],
                                  train_dataset['XS']), axis=1)
        
        test_X = np.concatenate ((test_dataset['XB'], 
                                  test_dataset['XD'], 
                                  test_dataset['XL'], 
                                  test_dataset['XM'], 
                                  test_dataset['XP'],
                                  test_dataset['XS']), axis=1)
        # Apply SelectKBest
        if feature_selection:
            train_X, test_X = feature_selection_SelectKBest(train_X, train_Y, test_X, k_best=num_features)

    elif data_type =="ZSF":
        train_X = np.concatenate((train_dataset['XB'], 
                                  train_dataset['XD'], 
                                  train_dataset['XL'], 
                                  train_dataset['XM'], 
                                  train_dataset['XP'],
                                  train_dataset['XS'],
                                  train_dataset['XF']), axis=1)
        
        test_X = np.concatenate ((test_dataset['XB'], 
                                  test_dataset['XD'], 
                                  test_dataset['XL'], 
                                  test_dataset['XM'], 
                                  test_dataset['XP'],
                                  test_dataset['XS'],
                                  test_dataset['XF']), axis=1)

        # Apply SelectKBest
        if feature_selection:
            train_X, test_X = feature_selection_SelectKBest(train_X, train_Y, test_X, k_best=num_features)
        
    elif data_type == 'ZSE':
        train_X1 = np.concatenate((train_dataset['XB'], 
                                  train_dataset['XD'], 
                                  train_dataset['XL'], 
                                  train_dataset['XM'], 
                                  train_dataset['XP'],
                                  train_dataset['XS']), axis=1)
        
        test_X1 = np.concatenate ((test_dataset['XB'], 
                                  test_dataset['XD'], 
                                  test_dataset['XL'], 
                                  test_dataset['XM'], 
                                  test_dataset['XP'],
                                  test_dataset['XS']), axis=1)
        if feature_selection:
            # Apply SelectKBest
            train_X2, test_X2 = feature_selection_SelectKBest(train_X1, train_Y, test_X1, k_best=num_features)

            # Apply SelectKBest_CHI2
            train_X3, test_X3 = feature_selection_SelectKBest_CHI2_BoW(train_dataset['XE'], train_Y, test_dataset['XE'])
        
            train_X = np.concatenate((train_X2, train_X3), axis=1)
            test_X  = np.concatenate((test_X2, test_X3), axis=1)
        else:
            train_X = np.concatenate((train_X1, train_dataset['XE']), axis=1)
            test_X  = np.concatenate((test_X1,  test_dataset['XE']), axis=1)

    elif data_type == 'ZSEF':
        train_X1 = np.concatenate((train_dataset['XB'], 
                                  train_dataset['XD'], 
                                  train_dataset['XL'], 
                                  train_dataset['XM'], 
                                  train_dataset['XP'],
                                  train_dataset['XS'],
                                  train_dataset['XF']), axis=1)
        
        test_X1 = np.concatenate ((test_dataset['XB'], 
                                  test_dataset['XD'], 
                                  test_dataset['XL'], 
                                  test_dataset['XM'], 
                                  test_dataset['XP'],
                                  test_dataset['XS'],
                                  test_dataset['XF']), axis=1)

        if feature_selection:
            # Apply SelectKBest
            train_X2, test_X2 = feature_selection_SelectKBest(train_X1, train_Y, test_X1, k_best=num_features)

            # Apply SelectKBest_CHI2
            train_X3, test_X3 = feature_selection_SelectKBest_CHI2_BoW(train_dataset['XE'], train_Y, test_dataset['XE'])
            
            train_X = np.concatenate((train_X2, train_X3), axis=1)
            test_X  = np.concatenate((test_X2, test_X3), axis=1)
        else:
            train_X = np.concatenate((train_X1, train_dataset['XE']), axis=1)
            test_X  = np.concatenate((test_X1,  test_dataset['XE']), axis=1)

    elif data_type =="E":
        if feature_selection:
            train_X, test_X = feature_selection_SelectKBest_CHI2_BoW(train_dataset['XE'], train_Y, test_dataset['XE'])
        else:
            train_X = np.array(train_dataset['XE'] )
            test_X  = np.array(test_dataset['XE'] )

    elif data_type =="F":
        train_X = np.array(train_dataset['XF'] )
        test_X  = np.array(test_dataset['XF'] )

        if feature_selection:
            # Apply SelectKBest
            train_X, test_X = feature_selection_SelectKBest(train_X, train_Y, test_X, k_best=num_features)

    return train_X, train_Y, test_X, test_Y



# Function to load data from pickle files
def load_pickle(filename):
    with open(filename, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict

def load_dict_from_pickle(filename):
    with open(filename, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict

def load_data_D(file_path, data_type, device):
    X = load_dict_from_pickle(f'{file_path}/{data_type}.pkl')
    Y = load_dict_from_pickle(f'{file_path}/Y.pkl')

    X = torch.tensor(X, dtype=torch.float).to(device)
    Y = torch.tensor(Y, dtype=torch.long).to(device)

    return X, Y

def load_data(file_path, data_type):
    X_train = np.array(load_dict_from_pickle(f'{file_path}/{data_type}_train.pkl'))
    X_test  = np.array(load_dict_from_pickle(f'{file_path}/{data_type}_test.pkl'))

    Y_train = np.array(load_dict_from_pickle(f'{file_path}/Y_train.pkl'))
    Y_test  = np.array(load_dict_from_pickle(f'{file_path}/Y_test.pkl'))

    return X_train, X_test, Y_train, Y_test


def save_metrics_to_csv(metrics, model_name, csv_file='ml_metrics.csv'):
    # Flatten the metrics dictionary with prefixed keys
    flat_metrics = {f"{stat}_{key}": value for stat, values in metrics.items() for key, value in values.items()}
    flat_metrics['model'] = model_name

    # Convert metrics to a DataFrame
    metrics_df = pd.DataFrame([flat_metrics])

    # Reorder the columns dynamically to match the desired pattern
    metric_keys = [key for key in metrics['average'].keys()]
    ordered_columns = ['model'] + [
        f"{stat}_{key}" for key in metric_keys for stat in ['average', 'std_dev']
    ]
    metrics_df = metrics_df[ordered_columns]

    # Extract directory from the file path
    directory = os.path.dirname(csv_file)

    # Ensure the directory exists
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")

    # Save to CSV
    if not os.path.isfile(csv_file):
        metrics_df.to_csv(csv_file, index=False)
    else:
        metrics_df.to_csv(csv_file, mode='a', header=False, index=False)

    print(f"Metrics saved to {csv_file}")


# FUNCTIONS
def save_list_as_pickle(L, given_path, file_name):
    # Ensure the directory exists
    if not os.path.exists(given_path):
        os.makedirs(given_path)
        print(f'\tDirectory created: {given_path}')
    
    # Save the list as a pickle file
    print(f'\tSaving to {given_path}/{file_name}.pkl')
    with open(os.path.join(given_path, f'{file_name}.pkl'), 'wb') as file:
        pickle.dump(L, file)
