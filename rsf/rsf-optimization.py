import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Optimal Train C-index: 0.934 with min_samples_split: 2, min_samples_leaf: 1
# Optimal Test C-index: 0.719 with min_samples_split: 2, min_samples_leaf: 4

def run_trees(df, surv_data, covariates, name):
    n_estimators = 550
    min_samples_split_list = np.arange(2, 20, dtype=int)
    min_samples_leaf_list = np.arange(5, 12, dtype=int)
    test_size = 0.2  # Use a fixed test size

    train_c_indices = np.zeros((len(min_samples_split_list), len(min_samples_leaf_list)))
    test_c_indices = np.zeros((len(min_samples_split_list), len(min_samples_leaf_list)))
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[covariates], surv_data, test_size=test_size, random_state=42)  

    for i, min_samples_split in enumerate(min_samples_split_list):
        for j, min_samples_leaf in enumerate(min_samples_leaf_list):
            # Fit the Random Survival Forest model with varying n_estimators
            rsf = RandomSurvivalForest(n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)
            rsf.fit(X_train, y_train)
            
            # Evaluate model performance on the training set
            train_c_index = concordance_index_censored(y_train['PFS_STATUS'], y_train['PFS_MONTHS'], rsf.predict(X_train))
            train_c_indices[i, j] = train_c_index[0]  # Store the train c-index
            
            # Evaluate model performance on the test set
            test_c_index = concordance_index_censored(y_test['PFS_STATUS'], y_test['PFS_MONTHS'], rsf.predict(X_test))
            test_c_indices[i, j] = test_c_index[0]  # Store the test c-index
            
            print(f"min_samples_split: {min_samples_split}, min_samples_leaf: {min_samples_leaf}, Test C-index: {test_c_index[0]:.3f}")

    # Find the optimal C-index and corresponding parameters
    max_train_c_index = np.max(train_c_indices)
    max_test_c_index = np.max(test_c_indices)
    optimal_train_params = np.unravel_index(np.argmax(train_c_indices), train_c_indices.shape)
    optimal_test_params = np.unravel_index(np.argmax(test_c_indices), test_c_indices.shape)

    optimal_train_split = min_samples_split_list[optimal_train_params[0]]
    optimal_train_leaf = min_samples_leaf_list[optimal_train_params[1]]
    optimal_test_split = min_samples_split_list[optimal_test_params[0]]
    optimal_test_leaf = min_samples_leaf_list[optimal_test_params[1]]

    print(f"Optimal Train C-index: {max_train_c_index:.3f} with min_samples_split: {optimal_train_split}, min_samples_leaf: {optimal_train_leaf}")
    print(f"Optimal Test C-index: {max_test_c_index:.3f} with min_samples_split: {optimal_test_split}, min_samples_leaf: {optimal_test_leaf}")

    # Create 3D plot for Train and Test C-index
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid for min_samples_split and min_samples_leaf
    X, Y = np.meshgrid(min_samples_split_list, min_samples_leaf_list)

    # Plot Train C-index
    ax.plot_surface(X, Y, train_c_indices.T, cmap='viridis', alpha=0.7)

    # Plot Test C-index
    ax.plot_surface(X, Y, test_c_indices.T, cmap='plasma', alpha=0.7)

    # Setting labels and title
    ax.set_xlabel('min_samples_split')
    ax.set_ylabel('min_samples_leaf')
    ax.set_zlabel('C-index')
    ax.set_title(f'{name} Train and Test C-index vs min_samples_split and min_samples_leaf - Test Size: {test_size}, Random State: 42')

    # Adding legend and grid
    ax.grid()

    plt.savefig(f'rsf-3d-{name}-train-test-c-index.png')
    # Show plot
    plt.show()

# Load the dataset
df = pd.read_csv('survival.csv')
df.drop(columns = ["PEMBROLIZUMAB", "ATEZOLIZUMAB", "NIVOLUMAB", "CURRENT_SMOKER", "FORMER_SMOKER", "NEVER_SMOKER"], inplace = True)
df.drop(columns = ["MET_DRIVER", "BRAF_DRIVER", "ARID1A_DRIVER"], inplace = True)

# Create structured array for survival analysis
surv_data = Surv.from_dataframe('PFS_STATUS', 'PFS_MONTHS', df)

run_trees(df, surv_data, df.columns.difference(['PFS_STATUS', 'PFS_MONTHS']), 'MSK MIND LUAD')