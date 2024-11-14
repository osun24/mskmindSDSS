import pandas as pd

df = pd.read_csv('lung_msk_mind_2020_clinical_data.tsv', sep='\t', comment='#')

print(df.head())

print(df.info())

# print columns with 90% non-missing
print(df.columns[df.isnull().mean() < 0.1])

# print info for 90% non-missing
print(df[df.columns[df.isnull().mean() < 0.1]].info())

# CATEGORICAL: 
categorical = ['']

# QUANTITATIVE:
quantitative = ['Age', 'Albumin']

# BINARY:
binary = ['']