import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'survival.csv'
data = pd.read_csv(file_path)

# Extract the relevant columns for Kaplan-Meier estimation
T = data['PFS_MONTHS']
E = data['PFS_STATUS']

# Replace the values in the 'OS_STATUS' column
# 0: Censored
# 1: Event
E = E.replace('0:LIVING', 0)
E = E.replace('1:DECEASED', 1)
print(E)

# Create a KaplanMeierFitter instance
kmf = KaplanMeierFitter(label = 'MSK MIND 2022')

# Fit the data into the model
kmf.fit(T, event_observed=E)

print(kmf.event_table)

plt.rcParams.update({'font.size': 14})
# Plot the Kaplan-Meier estimate
plt.figure(figsize=(10, 6))
kmf.plot_survival_function(ci_show=True)
plt.title('Kaplan-Meier Survival Curve (95% CI)')
plt.xlabel('Progression-Free Survival Time (Months)')
plt.ylabel('Progression-Free Survival Probability')
plt.grid(True)
add_at_risk_counts(kmf)
plt.tight_layout()
plt.savefig('km/MSKMIND-Overall-Kaplan-Meier.png')
plt.show()