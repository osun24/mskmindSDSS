import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_corr(df, name):
    # Compute the correlation matrix
    corr_matrix = df.corr()

    # Create a mask for the upper triangle (optional, to make the heatmap cleaner)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Set up the matplotlib figure
    plt.figure(figsize=(25, 18))

    # Draw the heatmap with the mask
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap="coolwarm", center=0, linewidths=0.5)

    # Show plot
    plt.title(f'Correlation Matrix {name}', fontsize = 30)

    plt.tight_layout()

    name = name.replace(' ', '-')
    plt.savefig(f'eda/correlation-all-{name}-2.png')
    plt.show()

surv = pd.read_csv('survival.csv')
create_corr(surv, 'MSK MIND LUAD')