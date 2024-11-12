import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Loading the Parkinson's dataset
data = pd.read_excel("data/parkinsons_data-VOICE-features.xlsx")

# Function to rename columns
def rename_columns(df):
    return df.rename(columns=lambda x: x.replace(':', '_').replace('(', '_').replace(')', ''))

# Renaming columns
data = rename_columns(data)

# Defining the columns
identifier_columns = ['name']
feature_columns = ['MDVP_Fo_Hz', 'MDVP_Fhi_Hz', 'MDVP_Flo_Hz', 'MDVP_Jitter_%', 'MDVP_Jitter_Abs', 'MDVP_RAP', 'MDVP_PPQ', 'Jitter_DDP', 'MDVP_Shimmer', 'MDVP_Shimmer_dB', 'Shimmer_APQ3', 'Shimmer_APQ5', 'MDVP_APQ', 'Shimmer_DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
target_column = ['status']

# Function to round values
def round_values(df):
    df['MDVP_Fo_Hz'] = df['MDVP_Fo_Hz'].round(3)
    df['MDVP_Fhi_Hz'] = df['MDVP_Fhi_Hz'].round(3)
    df['MDVP_Flo_Hz'] = df['MDVP_Flo_Hz'].round(3)
    df['MDVP_Jitter_%'] = df['MDVP_Jitter_%'].round(5)
    df['MDVP_Jitter_Abs'] = df['MDVP_Jitter_Abs'].apply(lambda x: '{:.8f}'.format(x))
    df['MDVP_RAP'] = df['MDVP_RAP'].round(5)
    df['MDVP_PPQ'] = df['MDVP_PPQ'].round(5)
    df['Jitter_DDP'] = df['Jitter_DDP'].round(5)
    df['MDVP_Shimmer'] = df['MDVP_Shimmer'].round(5)
    df['MDVP_Shimmer_dB'] = df['MDVP_Shimmer_dB'].round(3)
    df['Shimmer_APQ3'] = df['Shimmer_APQ3'].round(5)
    df['Shimmer_APQ5'] = df['Shimmer_APQ5'].round(5)
    df['MDVP_APQ'] = df['MDVP_APQ'].round(5)
    df['Shimmer_DDA'] = df['Shimmer_DDA'].round(5)
    df['NHR'] = df['NHR'].round(5)
    df['HNR'] = df['HNR'].round(3)
    return df

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

    # Rounding the values in train data
    train_continuous = round_values(train_data[feature_columns + target_column].copy())

    # Saving the train set for continuous data
    pd.concat([train_data[identifier_columns], train_continuous], axis=1).to_csv(f'data/continuous/parkinson_data-VOICE-features-train{i}.csv', index=False)

    # Discretizing each feature column and adding to train data
    train_discrete = train_data[feature_columns + target_column].copy()
    for column in feature_columns + target_column:
        train_discrete[column] = get_discretized_vector(train_discrete[column])

    # Saving the train set for discretized data
    pd.concat([train_data[identifier_columns], train_discrete], axis=1).to_csv(f'data/discrete/parkinson_data-VOICE-features-train{i}.csv', index=False)

# Rounding the values in test data
test_continuous = round_values(test_data[feature_columns + target_column].copy())

# Saving the test set for continuous data
pd.concat([test_data[identifier_columns], test_continuous], axis=1).to_csv('data/continuous/parkinson_data-VOICE-features-test.csv', index=False)

# Discretizing the test set
test_discrete = test_data[feature_columns + target_column].copy()
for column in feature_columns + target_column:
    test_discrete[column] = get_discretized_vector(test_discrete[column])

# Saving the test set for discretized data
pd.concat([test_data[identifier_columns], test_discrete], axis=1).to_csv('data/discrete/parkinson_data-VOICE-features-test.csv', index=False)

print("Data has been successfully processed, rounded, and discretized.")