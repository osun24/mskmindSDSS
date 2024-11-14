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
    n_estimators_list = np.linspace(40,100,6, dtype=int)
    learning_rate_list = np.linspace(0.18, 0.22, 100)
    
    
    train_c_indices = np.zeros((len(n_estimators_list), len(learning_rate_list)))
    test_c_indices = np.zeros((len(n_estimators_list), len(learning_rate_list)))
    
    test_size = 0.2
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[covariates], surv_data, test_size=test_size, random_state=42)
    
    for i, n_estimators in enumerate(n_estimators_list):
        for j, learning_rate in enumerate(learning_rate_list):
            gbm = GradientBoostingSurvivalAnalysis(
                loss="coxph",
                learning_rate=learning_rate,
                n_estimators=n_estimators,
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
            
            print(f"Number of Trees: {n_estimators}, Learning Rate: {learning_rate}, Train C-Index: {train_c_index[0]:3f}, Test C-index: {test_c_index[0]:.3f}")
            
    # Find the optimal C-index and corresponding parameters
    max_train_c_index = np.max(train_c_indices)
    max_test_c_index = np.max(test_c_indices)
    optimal_train_params = np.unravel_index(np.argmax(train_c_indices), train_c_indices.shape)
    optimal_test_params = np.unravel_index(np.argmax(test_c_indices), test_c_indices.shape)
    
    optimal_train_trees = n_estimators_list[optimal_train_params[0]]
    optimal_train_rate = learning_rate_list[optimal_train_params[1]]
    optimal_test_trees = n_estimators_list[optimal_test_params[0]]
    optimal_test_rate = learning_rate_list[optimal_test_params[1]]

    print(f"Optimal Train C-index: {max_train_c_index:.3f} with n_estimators: {optimal_train_trees}, learning_rate: {optimal_train_rate}")
    print(f"Optimal Test C-index: {max_test_c_index:.3f} with n_estimators: {optimal_test_trees}, learning_rate: {optimal_test_rate}")

    # List top 5 most optimal LEARNING RATES based on Test C-index
    top_5_indices = np.argpartition(test_c_indices, -5, axis=None)[-5:]
    print("Top 5 most optimal parameters based on Test C-index:")
    for idx in top_5_indices:
        n_trees = n_estimators_list[idx[0]]
        lr = learning_rate_list[idx[1]]
        c_index = test_c_indices[idx[0], idx[1]]
        print(f"n_estimators: {n_trees}, learning_rate: {lr}, Test C-index: {c_index:.3f}, Train C-index: {train_c_indices[idx[0], idx[1]]:.3f}")
    
    # Create 3D plot for Train and Test C-index
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid for n_estimators and learning_rate
    N, L = np.meshgrid(n_estimators_list, learning_rate_list)
    
    # Plot Train C-index
    ax.plot_surface(N, L, train_c_indices.T, cmap='viridis', alpha=0.7)
    
    # Plot Test C-index
    ax.plot_surface(N, L, test_c_indices.T, cmap='plasma', alpha=0.7)
    
    # Setting labels and title
    ax.set_xlabel('Number of Trees')
    ax.set_ylabel('Learning Rate')
    
    ax.set_zlabel('C-index')
    ax.set_title(f'{name} Train and Test C-index vs Number of Trees and Learning Rate - Test Size: {test_size}, Random State: 42')
    
    # Adding legend and grid
    ax.grid()
    
    plt.savefig(f'gbm-3d-{name}-train-test-c-index.png')
    plt.show()
    
# Load the dataset
df = pd.read_csv('survival.csv')

df.drop(columns = ["PEMBROLIZUMAB", "ATEZOLIZUMAB", "NIVOLUMAB", "CURRENT_SMOKER", "FORMER_SMOKER", "NEVER_SMOKER"], inplace = True)
df.drop(columns = ["MET_DRIVER", "BRAF_DRIVER", "ARID1A_DRIVER"], inplace = True)
covariates = df.columns.difference(['PFS_STATUS', 'PFS_MONTHS'])

# Create structured array for survival analysis
surv_data = Surv.from_dataframe('PFS_STATUS', 'PFS_MONTHS', df)

run_gbm_trees(df, surv_data, covariates, 'MSK MIND LUAD')
        
"""
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 500],
    'subsample': [0.5, 0.75, 1.0],
    'max_depth': [1, 3, 5]
}

gbm_cv = GradientBoostingSurvivalAnalysis(random_state=42)

grid_search = GridSearchCV(
    gbm_cv,
    param_grid,
    cv=5,
    scoring='neg_brier_score',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print(f"Best cross-validated C-index: {grid_search.best_score_:.3f}")"""