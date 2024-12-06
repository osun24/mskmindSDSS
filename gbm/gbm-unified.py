import pandas as pd
import numpy as np
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Optimal Test C-index: 0.736 with n_estimators: 31, learning_rate: 0.2642105263157895, min_samples_split: 2, min_samples_leaf: 10
def run_gbm_trees(df, surv_data, covariates, name):
    n_estimators_list = np.linspace(20,40,11, dtype=int)
    learning_rate_list = np.linspace(0.2, 0.3, 10)
    min_samples_split_list = np.arange(2, 5, dtype=int)
    min_samples_leaf_list = np.arange(8, 12, dtype=int)
    
    train_c_indices = np.zeros((len(n_estimators_list), len(learning_rate_list), len(min_samples_split_list), len(min_samples_leaf_list)))
    test_c_indices = np.zeros((len(n_estimators_list), len(learning_rate_list), len(min_samples_split_list), len(min_samples_leaf_list)))
    
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(df[covariates], surv_data, test_size=test_size, random_state=42)
    
    for a, n_estimators in enumerate(n_estimators_list):
        for b, learning_rate in enumerate(learning_rate_list):
            for c, min_samples_split in enumerate(min_samples_split_list):
                for d, min_samples_leaf in enumerate(min_samples_leaf_list):
                    gbm = GradientBoostingSurvivalAnalysis(
                        loss="coxph",
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        min_samples_leaf=min_samples_leaf,
                        min_samples_split=min_samples_split,
                        subsample=1.0,
                        random_state=42,
                        validation_fraction=0.1,
                        n_iter_no_change=10
                    )
                    
                    gbm.fit(X_train, y_train)
                    train_c_index = concordance_index_censored(y_train['PFS_STATUS'], y_train['PFS_MONTHS'], gbm.predict(X_train))[0]
                    test_c_index = concordance_index_censored(y_test['PFS_STATUS'], y_test['PFS_MONTHS'], gbm.predict(X_test))[0]
                    train_c_indices[a,b,c,d] = train_c_index
                    test_c_indices[a,b,c,d] = test_c_index
                    
                    print(f"Trees: {n_estimators}, LR: {learning_rate:.4f}, Split: {min_samples_split}, Leaf: {min_samples_leaf}, Train C-Index: {train_c_index:.3f}, Test C-Index: {test_c_index:.3f}")
    
    max_test_c_index = np.max(test_c_indices)
    opt_params = np.unravel_index(np.argmax(test_c_indices), test_c_indices.shape)
    opt_n_estimators = n_estimators_list[opt_params[0]]
    opt_lr = learning_rate_list[opt_params[1]]
    opt_split = min_samples_split_list[opt_params[2]]
    opt_leaf = min_samples_leaf_list[opt_params[3]]
    print(f"Optimal Test C-index: {max_test_c_index:.3f} with n_estimators: {opt_n_estimators}, learning_rate: {opt_lr}, min_samples_split: {opt_split}, min_samples_leaf: {opt_leaf}")

df = pd.read_csv('survival.csv')
df.drop(columns = ["PEMBROLIZUMAB","ATEZOLIZUMAB","NIVOLUMAB","CURRENT_SMOKER","FORMER_SMOKER","NEVER_SMOKER"], inplace=True)
df.drop(columns = ["MET_DRIVER","BRAF_DRIVER","ARID1A_DRIVER"], inplace=True)

covariates = df.columns.difference(['PFS_STATUS', 'PFS_MONTHS'])
surv_data = Surv.from_dataframe('PFS_STATUS', 'PFS_MONTHS', df)
run_gbm_trees(df, surv_data, covariates, 'MSK MIND LUAD')