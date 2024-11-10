import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
import os

# Load data (keeping your error handling)
data = pd.read_csv("data/dementia_data-MRI-features.csv", encoding="utf-8")

continuous_columns = ['MR Delay', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
identifier_columns = ['Subject ID', 'MRI ID', 'Group', 'M/F', 'Visit', 'Hand']

# Create directory structure
os.makedirs('data/continuous', exist_ok=True)
os.makedirs('data/discrete', exist_ok=True)

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize imputer and discretizer
imputer = SimpleImputer(strategy='mean')
discretizer = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='quantile')

for i, (train_index, test_index) in enumerate(kf.split(data), 1):
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]

    # Impute missing values
    train_continuous = pd.DataFrame(imputer.fit_transform(train_data[continuous_columns]), 
                                    columns=continuous_columns, index=train_data.index)
    test_continuous = pd.DataFrame(imputer.transform(test_data[continuous_columns]), 
                                   columns=continuous_columns, index=test_data.index)

    # Round values as per requirements
    for column in continuous_columns:
        if column in ['MR Delay', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV']:
            train_continuous[column] = train_continuous[column].round().astype(int)
            test_continuous[column] = test_continuous[column].round().astype(int)
        elif column == 'CDR':
            train_continuous[column] = train_continuous[column].apply(lambda x: int(x) if x.is_integer() else x)
            test_continuous[column] = test_continuous[column].apply(lambda x: int(x) if x.is_integer() else x)
        elif column in ['nWBV', 'ASF']:
            train_continuous[column] = train_continuous[column].round(3)
            test_continuous[column] = test_continuous[column].round(3)

    # Discretize continuous columns
    train_discrete = pd.DataFrame(discretizer.fit_transform(train_continuous), 
                                  columns=[f'{col}_discretized' for col in continuous_columns], 
                                  index=train_data.index)
    test_discrete = pd.DataFrame(discretizer.transform(test_continuous), 
                                 columns=[f'{col}_discretized' for col in continuous_columns], 
                                 index=test_data.index)

    # Save continuous data
    pd.concat([train_data[identifier_columns], train_continuous], axis=1).to_csv(f'data/continuous/dementia_data-MRI-features-train{i}.csv', index=False)
    
    # Save discrete data
    pd.concat([train_data[identifier_columns], train_discrete], axis=1).to_csv(f'data/discrete/dementia_data-MRI-features-train{i}.csv', index=False)

    # Save test data (only for the last fold)
    if i == 5:
        pd.concat([test_data[identifier_columns], test_continuous], axis=1).to_csv('data/continuous/dementia_data-MRI-features-test.csv', index=False)
        pd.concat([test_data[identifier_columns], test_discrete], axis=1).to_csv('data/discrete/dementia_data-MRI-features-test.csv', index=False)

print("Data has been successfully processed, imputed, and discretized.")