import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, brier_score_loss
from scipy.special import kl_div
import time

class ModelEvaluator:
    def __init__(self, configfile, datafile_test):
        self.configfile = configfile
        self.datafile_test = datafile_test
        self.running_time = 0
        self.inference_time = 0
        
        # Load configuration and test data
        self.load_config()
        self.load_test_data()
        
        # Evaluate model performance (dummy predictions for demonstration)
        self.evaluate_model()

    def load_config(self):
        """ Load the Bayesian network configuration from the specified file. """
        print(f"Loading configuration from {self.configfile}...")
        # Here you would implement logic to read the config file and set up your Bayesian network.
        # For now, we'll just print a message.
        
    def load_test_data(self):
        """ Load test data from the specified CSV file. """
        print(f"Loading test data from {self.datafile_test}...")
        try:
            self.test_data = pd.read_csv(self.datafile_test)
            print("Test data loaded successfully.")
            
            # Assuming 'Group' is the true label in your dataset.
            self.y_true = self.test_data['Group'].values
            
            # Generate dummy predictions (replace with actual model predictions)
            self.y_pred = np.random.choice([0, 1, 2], size=len(self.y_true))  # Randomly chosen classes
            
            # Generate dummy probabilities and normalize them
            random_probs = np.random.rand(len(self.y_true), 3)  # Random values for three classes
            self.y_prob = random_probs / random_probs.sum(axis=1, keepdims=True)  # Normalize to sum to 1

        except Exception as e:
            print(f"Error loading test data: {e}")
            sys.exit(1)

    def evaluate_model(self):
        """ Evaluate model performance based on loaded test data. """
        start_time = time.time()

        # Calculate metrics
        bal_acc = balanced_accuracy_score(self.y_true, self.y_pred)
        f1 = f1_score(self.y_true, self.y_pred, average='weighted')  # Use weighted average for multi-class
        auc = roc_auc_score(self.y_true, self.y_prob, multi_class='ovr')  # One-vs-Rest for multi-class AUC

        # Calculate Brier score for multiclass
        brier = np.mean([brier_score_loss(self.y_true == i, self.y_prob[:, i]) for i in range(3)])

        # KL Divergence: Normalize true labels to match probability distribution format
        true_distribution = np.zeros(3)  # For three classes (0, 1, 2)
        for label in self.y_true:
            true_distribution[int(label)] += 1
            
        true_distribution /= true_distribution.sum()  # Normalize to get probabilities
        
        kl_divergence = kl_div(true_distribution, np.mean(self.y_prob, axis=0)).sum()  # Compare with mean predicted probabilities

        # End timing the evaluation process
        self.running_time = time.time() - start_time
        self.inference_time = self.running_time  # Assuming inference time is same as evaluation time for simplicity

        # Print results
        print("\nCOMPUTING performance on test data...")
        print("Balanced Accuracy=" + str(bal_acc))
        print("F1 Score=" + str(f1))
        print("Area Under Curve=" + str(auc))
        print("Brier Score=" + str(brier))
        print("KL Divergence=" + str(kl_divergence))        
        print("Training Time=" + str(self.running_time) + " secs.")
        print("Inference Time=" + str(self.inference_time) + " secs.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: ModelEvaluator.py [config_file.txt] [test_file.csv]")
        print("EXAMPLE> ModelEvaluator.py config-lungcancer.txt lung_cancer-test.csv")
        exit(0)
    else:
        configfile = sys.argv[1]
        datafile_test = sys.argv[2]
        
    evaluator = ModelEvaluator(configfile, datafile_test)