import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# Assuming 'df' is your DataFrame with the data
# Define the survival time and event columns
surv = pd.read_csv('survival.csv')

# Exclude treatment because it is collinear with clinically reported PD-L1 score
# Don't include smoking status because it is collinear with smoking history
surv.drop(columns = ["PEMBROLIZUMAB", "ATEZOLIZUMAB", "NIVOLUMAB", "CURRENT_SMOKER", "FORMER_SMOKER", "NEVER_SMOKER"], inplace = True)

# List of variables to check
variables = [
    'EGFR_DRIVER', 'ERBB2_DRIVER', 'STK11_DRIVER', 'PACK-YEAR_HISTORY',
    'MSI_SCORE', 'FRACTION_GENOME_ALTERED', 'DNLR', 'IMPACT_TMB_SCORE',
    'CLINICALLY_REPORTED_PD-L1_SCORE', 'ALBUMIN', 'AGE', 'ECOG', 'IS_FEMALE'
]

log_vars = [
    "CLINICALLY_REPORTED_PD-L1_SCORE", "IMPACT_TMB_SCORE", "FRACTION_GENOME_ALTERED", "PACK-YEAR_HISTORY", "DNLR", "ALBUMIN"
]

# log transform variables
for var in log_vars:
    surv[var] = np.log(surv[var] + 1)

def check_proportional_hazards(df, var, T, E):
    print(f"Processing variable: {var}")
    
    if df[var].dtype == 'float64':
        # if pdl1 use above 50 as high and below 50 as low
        if var == 'CLINICALLY_REPORTED_PD-L1_SCORE':
            df['group'] = df[var].apply(lambda x: 'High' if x > 0 else 'Low')
            group_mapping = {'Low': '<50', 'High': '>=50'}
        else:
            # Continuous variable: divide into tertiles
            df['group'] = pd.qcut(df[var], q=3, labels=['Low', 'Medium', 'High'])
            # Create labels with ranges
            tertiles = df[var].quantile([0, 1/3, 2/3, 1]).values
            group_labels = ['{:.2f}-{:.2f}'.format(tertiles[i], tertiles[i+1]) for i in range(3)]
            group_mapping = dict(zip(['Low', 'Medium', 'High'], group_labels))
    else:
        # Categorical variable: use existing categories
        df['group'] = df[var]
        group_mapping = None
        
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10,6))
    
    # Plot the log(-log(survival)) curves for each group
    for name, grouped_df in df.groupby('group'):
        kmf.fit(grouped_df[T], event_observed=grouped_df[E], label=str(name))
        survival_function = kmf.survival_function_
        time = survival_function.index
        surv = survival_function[kmf._label]
        # Avoid log(0) or log(-log(1))
        surv = np.clip(surv, 1e-6, 1 - 1e-6)
        loglog_surv = np.log(-np.log(surv))
        plt.plot(np.log(time), loglog_surv, label=str(name))
    
    plt.xlabel('log(Time)')
    plt.ylabel('log(-log(Survival probability))')
    plt.title(f'Log(-Log(Survival)) plot for {var}')
    if group_mapping:
        handles, labels = plt.gca().get_legend_handles_labels()
        labels = [group_mapping.get(label, label) for label in labels]
        plt.legend(handles, labels, title='Group')
    else:
        plt.legend(title='Group')
        
    # save 
    plt.savefig(f'eda/loglog/log-log-{var}.png')
    plt.show()
    


# Run the function for each variable
for var in variables:
    check_proportional_hazards(surv, var, 'PFS_MONTHS', 'PFS_STATUS')