import pandas as pd
import numpy as np
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
import matplotlib.pyplot as plt
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn.model_selection import KFold

# Load the dataset
df = pd.read_csv('survival.csv')

df.drop(columns = ["PEMBROLIZUMAB", "ATEZOLIZUMAB", "NIVOLUMAB", "CURRENT_SMOKER", "FORMER_SMOKER", "NEVER_SMOKER"], inplace = True)
df.drop(columns = ["MET_DRIVER", "BRAF_DRIVER", "ARID1A_DRIVER"], inplace = True)
covariates = df.columns.difference(['PFS_STATUS', 'PFS_MONTHS'])

# Create structured array for survival analysis
surv_data = Surv.from_dataframe('PFS_STATUS', 'PFS_MONTHS', df)

test_size = 0.2

# Instantiate the model with desired hyperparameters
gbm = GradientBoostingSurvivalAnalysis(
    loss="coxph",
    learning_rate=0.2,
    n_estimators=50,
    subsample=1.0,
    random_state=42, 
    validation_fraction=0.1,
    n_iter_no_change=10,
    max_depth = 3,
    min_samples_split = 2,
    min_samples_leaf = 1
)

# Cross validation
# Prepare cross-validation strategy (e.g., 5-fold)
kf = KFold(n_splits=10)

# To store scores
scores = []

# Perform cross-validation manually
for train_index, test_index in kf.split(df[covariates]):
    X_train, X_test = df[covariates].iloc[train_index], df[covariates].iloc[test_index]
    y_train, y_test = surv_data[train_index], surv_data[test_index]
    
    # Export test/train folds to txt
    X_train.to_csv(f'X_train_fold{test_index[0]}.txt', index=False)
    
    
    # Fit the model on training data
    gbm.fit(X_train, y_train)
    
    # Predict risk scores for test set
    risk_scores = gbm.predict(X_test)
    
    # Calculate concordance index
    cindex = concordance_index_censored(y_test['PFS_STATUS'], y_test['PFS_MONTHS'], risk_scores)[0]
    
    # Store the score
    scores.append(cindex)
    
print(f"Mean C-Index: {np.mean(scores):.3f}")

# Plot the C-Index
plt.figure(figsize=(12,8))

plt.plot(scores, marker='o', linestyle='None')

plt.xlabel('Fold')
plt.ylabel('C-Index')
plt.title('C-Index for 5-Fold Cross Validation')
plt.grid()
plt.show()