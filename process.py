import pandas as pd

df = pd.read_csv('lung_msk_mind_2020_clinical_data.tsv', sep='\t', comment='#')

# Only keep columns with less than 10% missing values
df = df[df.columns[df.isnull().mean() < 0.1]]

# Rename all columns to all caps with spaces replaced by underscores
df.columns = [col.upper().replace(' ', '_') for col in df.columns]

"""
    #   Column                                                          Non-Null Count  Dtype  
---  ------                                                          --------------  -----  
 0   Study ID                                                        247 non-null    object 
 1   Patient ID                                                      247 non-null    object 
 2   Sample ID                                                       247 non-null    object 
 3   Age                                                             246 non-null    float64
 4   Age at Which Sequencing was Reported (Years)                    247 non-null    object 
 5   Albumin                                                         246 non-null    float64
 6   ALK driver                                                      246 non-null    object 
 7   Archer Panel                                                    247 non-null    object 
 8   ARID1A driver                                                   246 non-null    object 
 9   BOR                                                             246 non-null    object 
 10  BRAF driver                                                     246 non-null    object 
 11  Cancer Type                                                     247 non-null    object 
 12  Cancer Type Detailed                                            247 non-null    object 
 13  Clinically reported PD-L1 score                                 246 non-null    float64
 14  CT scan type                                                    246 non-null    object 
 15  Impact TMB Percentile (Across All Tumor Types)                  244 non-null    float64
 16  Impact TMB Score                                                244 non-null    float64
 17  Impact TMB Percentile (Within Tumor Type)                       244 non-null    float64
 18  Date added to cBioPortal                                        247 non-null    object 
 19  Disease                                                         246 non-null    object 
 20  dNLR                                                            246 non-null    float64
 21  Date of last drug administration                                246 non-null    float64
 22  Drug start date                                                 246 non-null    float64
 23  ECOG                                                            246 non-null    float64
 24  EGFR driver                                                     246 non-null    object 
 25  ERBB2 driver                                                    246 non-null    object 
 26  Date of Last Contact                                            246 non-null    float64
 27  Fraction Genome Altered                                         247 non-null    float64
 28  Gene Panel                                                      247 non-null    object 
 29  Histology                                                       246 non-null    object 
 30  Was IMPACT done on the same tissue that PD-L1 IHC was done on?  246 non-null    object 
 31  Institute Source                                                247 non-null    object 
 32  IO drug name                                                    246 non-null    object 
 33  Line of therapy                                                 246 non-null    float64
 34  MET driver                                                      246 non-null    object 
 35  Monotherapy vs. Combination                                     246 non-null    object 
 36  Month added to cBioPortal                                       247 non-null    object 
 37  MSI Score                                                       246 non-null    float64
 38  MSI Type                                                        246 non-null    object 
 39  Mutation Count                                                  242 non-null    float64
 40  Oncotree Code                                                   247 non-null    object 
 41  Pack-year history                                               246 non-null    object 
 42  MSK Pathology Slide Available                                   247 non-null    object 
 43  PD-L1 tissue site                                               246 non-null    object 
 44  PFS date                                                        246 non-null    float64
 45  PFS Months                                                      246 non-null    float64
 46  PFS Status                                                      246 non-null    object 
 47  Primary Tumor Site                                              247 non-null    object 
 48  RET driver                                                      246 non-null    object 
 49  ROS1 driver                                                     246 non-null    object 
 50  Sample Class                                                    247 non-null    object 
 51  Number of Samples Per Patient                                   247 non-null    int64  
 52  Sample coverage                                                 247 non-null    int64  
 53  Sample Type                                                     247 non-null    object 
 54  Sex                                                             246 non-null    object 
 55  What is the patient's smoking status?                           246 non-null    object 
 56  Somatic Status                                                  247 non-null    object 
 57  Status                                                          246 non-null    object 
 58  STK11 driver                                                    246 non-null    object 
 59  TMB                                                             246 non-null    float64
 60  Tumor Purity                                                    242 non-null    float64
 61  Treatment Setting                                               246 non-null    object 
 62  Is the patient deceased?                                        246 non-null    object 
 63  Week added to cBioPortal                                        247 non-null    object  
"""

# Drop those with missing PFS_MONTHS
df = df.dropna(subset=['PFS_MONTHS'])

# Drop those with missing/unknown PACK-YEAR_HISTORY (n = 7)
df = df[~df['PACK-YEAR_HISTORY'].isin(['unk', 'NA', 'Cigars'])]
df['PACK-YEAR_HISTORY'] = df['PACK-YEAR_HISTORY'].astype(float)

# Drop those in clinical trial treatment setting
df = df[~df['TREATMENT_SETTING'].isin(['Clinical trial'])]

binary = ["ALK_DRIVER", "ARID1A_DRIVER", "BRAF_DRIVER", "EGFR_DRIVER", "ERBB2_DRIVER", "MET_DRIVER", "RET_DRIVER", "ROS1_DRIVER", "STK11_DRIVER"]

parameters = binary + ["PATIENT_ID", "PFS_MONTHS", "PFS_STATUS","WHAT_IS_THE_PATIENT'S_SMOKING_STATUS?", "MONOTHERAPY_VS._COMBINATION", "SEX", "PACK-YEAR_HISTORY", "MSI_SCORE", "IO_DRUG_NAME", "FRACTION_GENOME_ALTERED", "DNLR", "IMPACT_TMB_SCORE", "CLINICALLY_REPORTED_PD-L1_SCORE", "ALBUMIN", "AGE", "ECOG"]

print(f"Parameters: {len(parameters)}")

df = df[parameters]

print(df.head())

# Convert binary TRUE/FALSE to 0/1
for col in binary:
    df[col] = df[col].astype(str).str.lower().map({'true': 1, 'false': 0})
    print(df[col].value_counts())
    
# Drop ALK, RET, ROS1 because all are FALSE
# Combination therapy for those with PD1+CTLA4 --> particularly treatments like 
df = df.drop(columns=['ALK_DRIVER', 'RET_DRIVER', "ROS1_DRIVER", "MONOTHERAPY_VS._COMBINATION"])

# Convert SEX to IS_FEMALE (majority - ~54%)
df['IS_FEMALE'] = df['SEX'].map({'Female': 1, 'Male':0})
df.drop(columns=['SEX'], inplace=True)

# map PFS_STATUS to 0/1
df['PFS_STATUS'] = df['PFS_STATUS'].map({'1:PROGRESS': 1, '0:CENSORED': 0})

# rename "WHAT IS THE PATIENT'S SMOKING STATUS?" to SMOKING_STATUS
df.rename(columns={"WHAT_IS_THE_PATIENT'S_SMOKING_STATUS?": "SMOKING_STATUS"}, inplace=True)

# Set dummies for categorical variables
df = pd.get_dummies(df, columns=["SMOKING_STATUS"], drop_first=False)

# Dummy IO drug name into separate columns - "Nivolumab", "Pembrolizumab", "Atezolizumab"
# Drop with special IO drug treatment - n = 14
df = df[~df['IO_DRUG_NAME'].isin(['Resection + Nivolumab', 'Nivolumab (+added ipilimumab)', 'Durvalumab+Tremelimumab', 'Resection + Ipilimumab+Nivolumab', "Ipilimumab+Nivolumab"])]
print(df['IO_DRUG_NAME'].value_counts())
df = pd.get_dummies(df, columns=['IO_DRUG_NAME'], drop_first=False)

# Rename columns: SMOKING_STATUS_Former smoker, SMOKING_STATUS_Never smoker, SMOKING_STATUS_Smoker
df.rename(columns={"SMOKING_STATUS_Former smoker": "FORMER_SMOKER", "SMOKING_STATUS_Never smoker": "NEVER_SMOKER", "SMOKING_STATUS_Current smoker (within 6 months of diagnosis)": "CURRENT_SMOKER"}, inplace=True)
df.rename(columns = {"IO_DRUG_NAME_Atezolizumab": "ATEZOLIZUMAB", "IO_DRUG_NAME_Nivolumab": "NIVOLUMAB", "IO_DRUG_NAME_Pembrolizumab": "PEMBROLIZUMAB"}, inplace=True)

# drop missing values - n = 5
df = df.dropna()

print(df.info())

# number of unique patient ids
print(df['PATIENT_ID'].nunique())
df.drop(columns=['PATIENT_ID'], inplace=True)

# export to csv
df.to_csv('survival.csv', index=False)