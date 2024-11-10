import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Loading the Parkinson's dataset
data = pd.read_excel("data\parkinsons_data-VOICE-features.xlsx")

# Function to rename columns
def rename_columns(df):
    return df.rename(columns=lambda x: x.replace(':', '_').replace('(', '_').replace(')', ''))

# Renaming columns
data = rename_columns(data)

# Defining the columns
identifier_columns = ['name']
feature_columns = ['MDVP_Fo_Hz', 'MDVP_Fhi_Hz', 'MDVP_Flo_Hz', 'MDVP_Jitter_%', 'MDVP_Jitter_Abs', 'MDVP_RAP', 'MDVP_PPQ', 'Jitter_DDP', 'MDVP_Shimmer', 'MDVP_Shimmer_dB', 'Shimmer_APQ3', 'Shimmer_APQ5', 'MDVP_APQ', 'Shimmer_DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
target_column = ['status']

# Function to impute missing values with mean
def mean_impute(column):
    return column.fillna(column.mean())

# Imputing missing values for all numeric columns
numeric_columns = feature_columns + target_column
for column in numeric_columns:
    if data[column].dtype in ['int64', 'float64']:
        data[column] = mean_impute(data[column])

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

    # Saving the train set for continuous data
    train_data[identifier_columns + feature_columns + target_column].to_csv(f'data/continuous/parkinson_data-VOICE-features-train{i}.csv', index=False)

    # Discretizing each feature column and adding to train data
    for column in feature_columns + target_column:
        train_data[f'{column}_discretized'] = get_discretized_vector(train_data[column])

    # Saving the train set for discretized data
    train_data[identifier_columns + [f'{col}_discretized' for col in feature_columns + target_column]].to_csv(f'data/discrete/parkinson_data-VOICE-features-train{i}.csv', index=False)

# Discretizing the test set
for column in feature_columns + target_column:
    test_data[f'{column}_discretized'] = get_discretized_vector(test_data[column])

# Saving the test set for continuous data
test_data[identifier_columns + feature_columns + target_column].to_csv('data/continuous/parkinson_data-VOICE-features-test.csv', index=False)

# Saving the test set for discretized data
test_data[identifier_columns + [f'{col}_discretized' for col in feature_columns + target_column]].to_csv('data/discrete/parkinson_data-VOICE-features-test.csv', index=False)

print("Data has been successfully processed and saved with renamed columns.")