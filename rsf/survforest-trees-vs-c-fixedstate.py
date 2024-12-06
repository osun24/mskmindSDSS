import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

def run_trees(df, surv_data, covariates, name):
    # Array to store results
    n_estimators_list = np.linspace(10, 1000, 25, dtype=int)
    test_size = 0.2  # Use a fixed test size

    train_c_indices = []
    test_c_indices = []

    for n_estimators in n_estimators_list:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df[covariates], surv_data, test_size=test_size, random_state=42)  
        
        # Fit the Random Survival Forest model with varying n_estimators
        rsf = RandomSurvivalForest(n_estimators=n_estimators, min_samples_split=10, min_samples_leaf=8, random_state=42)
        rsf.fit(X_train, y_train)
        
        # Evaluate model performance on the training set
        train_c_index = concordance_index_censored(y_train['PFS_STATUS'], y_train['PFS_MONTHS'], rsf.predict(X_train))
        train_c_indices.append(train_c_index[0])  # Store the training c-index
        
        # Evaluate model performance on the test set
        test_c_index = concordance_index_censored(y_test['PFS_STATUS'], y_test['PFS_MONTHS'], rsf.predict(X_test))
        test_c_indices.append(test_c_index[0])  # Store the test c-index
        
        print(f"Number of Trees: {n_estimators}, Test C-index: {test_c_index[0]:.3f}")

    # Plot Train and Test C-index with Confidence Intervals
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_list, train_c_indices, label='Train C-index')
    plt.plot(n_estimators_list, test_c_indices, label='Test C-index')
    plt.xlabel('Number of Trees')
    plt.ylabel('C-index')
    plt.title(f'{name} Train and Test C-index with 95% CI vs Number of Trees - Test Size: {test_size}, Random State: 42')
    plt.legend()
    plt.grid()
    plt.savefig(f'rsf-{name}-numtrees-vs-c-testsize{test_size}-state42.png')
    plt.show()

# Load the dataset
df = pd.read_csv('survival.csv')
df.drop(columns = ["PEMBROLIZUMAB", "ATEZOLIZUMAB", "NIVOLUMAB", "CURRENT_SMOKER", "FORMER_SMOKER", "NEVER_SMOKER"], inplace = True)
df.drop(columns = ["MET_DRIVER", "BRAF_DRIVER", "ARID1A_DRIVER"], inplace = True)

# Create structured array for survival analysis
surv_data = Surv.from_dataframe('PFS_STATUS', 'PFS_MONTHS', df)

run_trees(df, surv_data, df.columns.difference(['PFS_STATUS', 'PFS_MONTHS']), 'MSK MIND LUAD')