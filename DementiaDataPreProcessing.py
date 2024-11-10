import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Loading the dataset
data = pd.read_csv(r'C:\Users\envy\Desktop\AI Coursework\data\dementia_data-MRI-features.csv')

# Defining the continuous columns
continuous_columns = ['MR Delay', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
identifier_columns = ['Subject ID', 'MRI ID', 'Group', 'M/F', 'Visit', 'Hand']

# Function to impute missing values with mean
def mean_impute(column):
    return column.fillna(column.mean())

# Imputing missing values for continuous columns
for column in continuous_columns:
    data[column] = mean_impute(data[column]).round(2)

# Function to discretize a vector
def get_discretized_vector(X):
    _mean = np.mean(X)
    _std = np.std(X)
    bins = [float('-inf'), _mean - (_std * 2), _mean - _std, _mean, _mean + _std, _mean + (_std * 2), float('inf')]
    return pd.cut(X, bins=bins, labels=[0, 1, 2, 3, 4, 5])

# Creating directory structure if it doesn't exist
os.makedirs('data/continuous', exist_ok=True)
os.makedirs('data/discrete', exist_ok=True)

# Preparing to create multiple train sets and one test set
test_data = None
for i in range(1, 6):  # Creating 5 train sets
    if test_data is None:
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    else:
        train_data, _ = train_test_split(data, test_size=0.2, random_state=i)

    # Saving the train set for continuous data with specified columns only
    train_data[identifier_columns + continuous_columns].to_csv(f'data/continuous/dementia_data-MRI-features-train{i}.csv', index=False)

    # Discretizing each continuous column and adding to train data
    for column in continuous_columns:
        train_data[f'{column}_discretized'] = get_discretized_vector(train_data[column])

    # Saving the train set for discretized data with specified columns only
    train_data[identifier_columns + [f'{col}_discretized' for col in continuous_columns]].to_csv(f'data/discrete/dementia_data-MRI-features-train{i}.csv', index=False)

# Discretizing the test set
for column in continuous_columns:
    test_data[f'{column}_discretized'] = get_discretized_vector(test_data[column])

# Saving the test set for continuous data with specified columns only
test_data[identifier_columns + continuous_columns].to_csv('data/continuous/dementia_data-MRI-features-test.csv', index=False)

# Saving the test set for discretized data with specified columns only
test_data[identifier_columns + [f'{col}_discretized' for col in continuous_columns]].to_csv('data/discrete/dementia_data-MRI-features-test.csv', index=False)

print("Data has been successfully processed and missing values have been imputed.")