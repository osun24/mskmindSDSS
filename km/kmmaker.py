import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts

# Load the data
df = pd.read_csv('survival-nodummies.csv')

# Define duration and event columns
duration_col = 'PFS_MONTHS'
event_col = 'PFS_STATUS'

# Rename "Current smoker (within 6 months of diagnosis)" to "Current smoker"
df['SMOKING_STATUS'] = df['SMOKING_STATUS'].replace('Current smoker (within 6 months of diagnosis)', 'Current smoker')

# Rename "IO_DRUG_NAME" to "IMMUNOTHERAPY"
df['IMMUNOTHERAPY'] = df['IO_DRUG_NAME']
df.drop('IO_DRUG_NAME', axis=1, inplace=True)

# List of categorical covariates (excluding quantitative variables)
categorical_covariates = ['ARID1A_DRIVER', 'BRAF_DRIVER', 'EGFR_DRIVER', 'ERBB2_DRIVER', 'MET_DRIVER',  'STK11_DRIVER', 'SMOKING_STATUS', 'IMMUNOTHERAPY']

def create_stage_variable(df, stage_prefix):
    """
    Combine stage indicator columns into a single categorical stage variable.
    """
    stage_columns = [col for col in df.columns if col.startswith(stage_prefix)]
    def get_stage(row):
        for col in stage_columns:
            if row[col] == 1:
                return col.replace(stage_prefix, '').strip()
        return np.nan
    df[stage_prefix.rstrip('_')] = df.apply(get_stage, axis=1)
    return df

def plot_km_and_logrank(df, duration_col, event_col, group_col):
    """
    Plot Kaplan-Meier curves and perform log-rank test for the given group column.
    Include p-value and number at risk table in the plot.
    """
    # Drop missing values for the group column
    df_plot = df.dropna(subset=[group_col, duration_col, event_col])
    groups = df_plot[group_col].unique()
    kmf_dict = {}
    
    # Prepare the figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for group in groups:
        ix = df_plot[group_col] == group
        kmf = KaplanMeierFitter()
        kmf.fit(df_plot.loc[ix, duration_col], event_observed=df_plot.loc[ix, event_col], label=str(group))
        kmf_dict[group] = kmf
        kmf.plot_survival_function(ax=ax, ci_show=True)
    
    # Perform log-rank test
    results = multivariate_logrank_test(df_plot[duration_col], df_plot[group_col], df_plot[event_col])
    p_value = results.p_value
    
    # Include p-value as text in the plot
    plt.title(f'Kaplan-Meier Survival Curves by {group_col}')
    plt.xlabel('Time (Months)')
    plt.ylabel('Progression-Free Survival Probability')
    plt.legend(title=group_col)
    plt.text(0.6, 0.1, f'Log-rank p-value: {p_value:.4f}', transform=ax.transAxes)
    
    # Add number at risk table
    add_at_risk_counts(*kmf_dict.values(), ax=ax)
    
    # Adjust layout to accommodate at-risk table
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f'km_plot_{group_col}.png')
    plt.close()
    
    return p_value

def km_overall(df, duration_col, event_col):
    # Create a KaplanMeierFitter instance
    kmf = KaplanMeierFitter(label = 'MSK MIND LUAD')

    # Fit the data into the model
    kmf.fit(df[duration_col], event_observed=df[event_col])

    print(kmf.event_table)

    # Plot the Kaplan-Meier estimate
    plt.figure(figsize=(10, 6))
    kmf.plot_survival_function(ci_show=True)
    plt.title('Kaplan-Meier Survival Curve (95% CI)')
    plt.xlabel('Progression-Free Survival Time (Months)')
    plt.ylabel('Survival Probability')
    plt.grid(True)
    add_at_risk_counts(kmf)
    plt.tight_layout()
    plt.savefig('km/MSKMETO-Overall-Kaplan-Meier.png')
    plt.show()
    
def main():
    for var in categorical_covariates:
        print(f'Processing variable: {var}')
        group_col = var
        p_value = plot_km_and_logrank(df, duration_col, event_col, group_col)
        print(f'Log-rank test p-value for {var}: {p_value:.4f}\n')
    km_overall(df, duration_col, event_col)

if __name__ == "__main__":
    main()