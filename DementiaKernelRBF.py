import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load data from the CSV files
train_file_path = 'data/continuous/dementia_data-MRI-features-train1.csv'
test_file_path = 'data/continuous/dementia_data-MRI-features-test.csv'

# Read the train and test datasets
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Convert 'MMSE' to numeric in both train and test datasets, setting non-numeric values to NaN
train_data['CDR'] = pd.to_numeric(train_data['CDR'], errors='coerce')
test_data['CDR'] = pd.to_numeric(test_data['CDR'], errors='coerce')

# Drop rows where 'CDR' has NaN values in both train and test datasets
train_data = train_data.dropna(subset=['CDR'])
test_data = test_data.dropna(subset=['CDR'])

# Limit the number of samples (for example, take the first 50 rows for both train and test)
train_sample_size = 5
test_sample_size = 2

# If the dataset has more than the desired number of samples, limit it
X_train = train_data[['CDR']].head(train_sample_size).values
X_test = test_data[['CDR']].head(test_sample_size).values

# Set the gamma value for the RBF kernel
gamma = 0.5

# Function to compute RBF kernel vector
def rbf_kernel_vector(x_test, X_train, gamma):
    # Calculate the kernel values for a single test sample against all train samples
    kernel_vector = np.array([
        np.exp(-gamma * np.sum((x_test - x_train) ** 2))  # Apply RBF formula
        for x_train in X_train
    ])
    return kernel_vector

# Calculate kernel vectors for each test sample
kernel_vectors = []
for x_test in X_test:
    kernel_vector = rbf_kernel_vector(x_test, X_train, gamma)
    kernel_vectors.append(kernel_vector)

# Convert kernel_vectors list to a 2D array for heatmap plotting
kernel_matrix = np.array(kernel_vectors)

# Visualization with improved settings
plt.figure(figsize=(12, 8))  # Larger figure size for readability
ax = sns.heatmap(kernel_matrix, 
                 annot=True, 
                 cmap='plasma',   # Use a different colormap for better contrast
                 square=True, 
                 fmt=".3f",       # Limit annotation to 3 decimal places
                 annot_kws={"size": 10},  # Set font size for annotations
                 cbar_kws={"label": 'Kernel Value'})  # Add label to the color bar

plt.title("RBF Kernel Vectors (CDR)", fontsize=16)
plt.xlabel("X_train samples", fontsize=14)
plt.ylabel("X_test samples", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()
