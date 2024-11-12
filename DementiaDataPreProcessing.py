import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing._discretization")

# Load data
data = pd.read_csv("data/dementia_data-MRI-features.csv", encoding="utf-8")

# Replace spaces with underscores in column names
data.columns = data.columns.str.replace(' ', '_')

continuous_columns = ['MR_Delay', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
categorical_columns = ['Group', 'M/F', 'Hand']
identifier_columns = ['Subject_ID', 'MRI_ID', 'Visit']
discrete_columns = ['MR_Delay', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV']

# Create directory structure
os.makedirs('data/continuous', exist_ok=True)
os.makedirs('data/discrete', exist_ok=True)

# Initialize KFold, imputer, and discretizer
kf = KFold(n_splits=5, shuffle=True, random_state=42)
imputer = SimpleImputer(strategy='mean')
n_bins = 4
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')

def encode_categorical(data):
    encoded = data.copy()
    encoded['Group'] = encoded['Group'].map({'Nondemented': 0, 'Demented': 1, 'Converted': 2})
    encoded['M/F'] = encoded['M/F'].map({'F': 0, 'M': 1})
    encoded['Hand'] = encoded['Hand'].map({'R': 0, 'L': 1})
    return encoded

for i, (train_index, test_index) in enumerate(kf.split(data), 1):
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]

    # Encode categorical variables
    train_data = encode_categorical(train_data)
    test_data = encode_categorical(test_data)

    # Impute missing values
    train_continuous = pd.DataFrame(imputer.fit_transform(train_data[continuous_columns]), 
                                    columns=continuous_columns, index=train_data.index)
    test_continuous = pd.DataFrame(imputer.transform(test_data[continuous_columns]), 
                                   columns=continuous_columns, index=test_data.index)

    # Round values as per requirements
    for column in continuous_columns:
        if column in ['MR_Delay', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV']:
            train_continuous[column] = train_continuous[column].round().astype(int)
            test_continuous[column] = test_continuous[column].round().astype(int)
        elif column == 'CDR':
            train_continuous[column] = train_continuous[column].apply(lambda x: int(x) if x.is_integer() else x)
            test_continuous[column] = test_continuous[column].apply(lambda x: int(x) if x.is_integer() else x)
        elif column in ['nWBV', 'ASF']:
            train_continuous[column] = train_continuous[column].round(3)
            test_continuous[column] = test_continuous[column].round(3)

    # Discretize continuous columns
    train_discrete = train_continuous.copy()
    test_discrete = test_continuous.copy()
    train_discrete[discrete_columns] = discretizer.fit_transform(train_continuous[discrete_columns])
    test_discrete[discrete_columns] = discretizer.transform(test_continuous[discrete_columns])

    # Add encoded categorical columns to both continuous and discrete datasets
    for cat_col in categorical_columns:
        train_continuous[cat_col] = train_data[cat_col]
        test_continuous[cat_col] = test_data[cat_col]
        train_discrete[cat_col] = train_data[cat_col]
        test_discrete[cat_col] = test_data[cat_col]

    # Save continuous data
    pd.concat([train_data[identifier_columns], train_continuous], axis=1).to_csv(f'data/continuous/dementia_data-MRI-features-train{i}.csv', index=False)
    
    # Save discrete data
    pd.concat([train_data[identifier_columns], train_discrete], axis=1).to_csv(f'data/discrete/dementia_data-MRI-features-train{i}.csv', index=False)

    # Save test data (only for the last fold)
    if i == 5:
        pd.concat([test_data[identifier_columns], test_continuous], axis=1).to_csv('data/continuous/dementia_data-MRI-features-test.csv', index=False)
        pd.concat([test_data[identifier_columns], test_discrete], axis=1).to_csv('data/discrete/dementia_data-MRI-features-test.csv', index=False)

print("Data has been successfully processed, imputed, and discretized.")