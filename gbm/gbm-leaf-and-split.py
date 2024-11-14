import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sksurv.metrics import concordance_index_censored
import joblib
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

def run_gbm_trees(df, surv_data, covariates, name):
    min_samples_split_list = np.arange(2, 10, dtype=int)
    min_samples_leaf_list = np.arange(1, 12, dtype=int)

    train_c_indices = np.zeros((len(min_samples_split_list), len(min_samples_leaf_list)))
    test_c_indices = np.zeros((len(min_samples_split_list), len(min_samples_leaf_list)))
    
    test_size = 0.2
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[covariates], surv_data, test_size=test_size, random_state=42)
    
    for i, min_samples_split in enumerate(min_samples_split_list):
        for j, min_samples_leaf in enumerate(min_samples_leaf_list):
            gbm = GradientBoostingSurvivalAnalysis(
                loss="coxph",
                n_estimators= 50,
                learning_rate = 0.2,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                subsample=1.0,
                random_state=42, 
                validation_fraction=0.1,
                n_iter_no_change=10
            )
            
            # Fit the model
            gbm.fit(X_train, y_train)
            
            # Evaluate model performance on the training set
            train_c_index = concordance_index_censored(y_train['PFS_STATUS'], y_train['PFS_MONTHS'], gbm.predict(X_train))
            train_c_indices[i, j] = train_c_index[0]  # Store the training c-index  
            
            # Evaluate model performance on the test set
            test_c_index = concordance_index_censored(y_test['PFS_STATUS'], y_test['PFS_MONTHS'], gbm.predict(X_test))
            test_c_indices[i, j] = test_c_index[0]  # Store the test c-index
            
            print(f"Min Samples SPLIT: {min_samples_split}, Min Samples LEAF: {min_samples_leaf}, Train C-Index: {train_c_index[0]:3f}, Test C-index: {test_c_index[0]:.3f}")
            
    # Find the optimal C-index and corresponding parameters
    max_train_c_index = np.max(train_c_indices)
    max_test_c_index = np.max(test_c_indices)
    optimal_train_params = np.unravel_index(np.argmax(train_c_indices), train_c_indices.shape)
    optimal_test_params = np.unravel_index(np.argmax(test_c_indices), test_c_indices.shape)
    
    optimal_train_trees = min_samples_split_list[optimal_train_params[0]]
    optimal_train_rate = min_samples_leaf_list[optimal_train_params[1]]
    optimal_test_trees = min_samples_split_list[optimal_test_params[0]]
    optimal_test_rate = min_samples_leaf_list[optimal_test_params[1]]

    print(f"Optimal Train C-index: {max_train_c_index:.3f} with min_samples_split: {optimal_train_trees}, min_samples_leaf: {optimal_train_rate}")
    print(f"Optimal Test C-index: {max_test_c_index:.3f} with min_samples_split: {optimal_test_trees}, min_samples_leaf: {optimal_test_rate}")

    # Create 3D plot for Train and Test C-index
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid for min_samples_split and min_samples_leaf
    N, L = np.meshgrid(min_samples_split_list, min_samples_leaf_list)
    
    # Plot Train C-index
    ax.plot_surface(N, L, train_c_indices.T, cmap='viridis', alpha=0.7)
    
    # Plot Test C-index
    ax.plot_surface(N, L, test_c_indices.T, cmap='plasma', alpha=0.7)
    
    # Setting labels and title
    ax.set_xlabel('Min Samples per Split')
    ax.set_ylabel('Min Samples per Leaf')
    
    ax.set_zlabel('C-index')
    ax.set_title(f'{name} Train and Test C-index with 95% CI vs Min Samples per Split and Min Samples per Leaf - Test Size: {test_size}, Random State: 42')
    
    # Adding legend and grid
    ax.grid()
    
    plt.savefig(f'gbm-3d-leaf-split-{name}-train-test-c-index.png')
    plt.show()
    
# Load the dataset
df = pd.read_csv('survival.csv')

df.drop(columns = ["PEMBROLIZUMAB", "ATEZOLIZUMAB", "NIVOLUMAB", "CURRENT_SMOKER", "FORMER_SMOKER", "NEVER_SMOKER"], inplace = True)
df.drop(columns = ["MET_DRIVER", "BRAF_DRIVER", "ARID1A_DRIVER"], inplace = True)
covariates = df.columns.difference(['PFS_STATUS', 'PFS_MONTHS'])

# Create structured array for survival analysis
surv_data = Surv.from_dataframe('PFS_STATUS', 'PFS_MONTHS', df)

run_gbm_trees(df, surv_data, covariates, 'MSK MIND LUAD')