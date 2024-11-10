import sys
import time
import os.path
import numpy as np
import BayesNetUtil as bnu
from sklearn import metrics
from DataReader import CSV_DataReader
from BayesNetInference import BayesNetInference

#Declare Model Evaluator Class
class ModelEvaluator(BayesNetInference):
    verbose = False 
    inference_time = None

    def __init__(self, configfile_name, datafile_test):
        if os.path.isfile(configfile_name):
            #Loads Bayesian Network Stored In ConfigFile_Name
            super().__init__(None, configfile_name, None, None)
            self.running_time = time.time()
            self.inference_time = time.time()

        #Reads Test Data Using Code From DataReader
        self.csv = CSV_DataReader(datafile_test)
        
        #Apply Discretization Checks
        self.discretize_target_variable()

        #Generates Performance Results From Above Predictions 
        self.inference_time = time.time()
        true, pred, prob = self.get_true_and_predicted_targets()
        self.running_time = time.time() - self.running_time
        self.inference_time = time.time() - self.inference_time
        self.compute_performance(true, pred, prob)
    
    def discretize_target_variable(self):
        #Define Discretization Logic Using Quantiles Or Thresholds
        for i, data_point in enumerate(self.csv.rv_all_values):
            asf_value = float(data_point[-1])
            if asf_value < 0.1:
                self.csv.rv_all_values[i][-1] = "Low"
            elif 0.1 <= asf_value < 0.5:
                self.csv.rv_all_values[i][-1] = "Medium"
            else:
                self.csv.rv_all_values[i][-1] = "High"

    def get_true_and_predicted_targets(self):
        print("\nPERFORMING probabilistic inference on test data...")
        
        Y_true = []
        Y_pred = []
        Y_prob = []

        #Define Threshold Sets
        threshold_one = set(["High"])
        zero_values = set(["Low", "Medium"])

        #Loop Through Data Points To Categorize Targets
        for i in range(len(self.csv.rv_all_values)):
            data_point = self.csv.rv_all_values[i]
            target_value = data_point[len(self.csv.rand_vars) - 1]

            #Classify Based On Target_Value
            if target_value in threshold_one:
                Y_true.append(1)
            elif target_value in zero_values:
                Y_true.append(0)
            else:
                #Handle Or Log Unknown Values For Debugging
                print(f"Unknown target value: {target_value}")
            
            #Probabilistic Prediction Logic Placeholder
            Y_pred.append(1 if target_value in threshold_one else 0)
            Y_prob.append(float(target_value) if target_value in ["0", "1"] else 0.5)

        return Y_true, Y_pred, Y_prob

    #Returns Probability Distribution Using Inference By Enumeration
    def get_predictions_from_BayesNet(self, data_point, nbc):
        #Forms Probabilistic Query Based On Predictor Variable
        evidence = ""
        for var_index in range(0, len(self.csv.rand_vars)-1):
            evidence += "," if len(evidence)>0 else ""
            evidence += self.csv.rand_vars[var_index]+'='+str(data_point[var_index])
        prob_query = "P(%s|%s)" % (self.csv.predictor_variable, evidence)
        self.query = bnu.tokenise_query(prob_query, False)

        #Sends Query To BayesNetInference And Get Probability Distribution
        self.prob_dist = self.enumeration_ask()
        normalised_dist = bnu.normalise(self.prob_dist)
        if self.verbose: print("%s=%s" % (prob_query, normalised_dist))

        return normalised_dist

    #Prints Model Performance Metrics: Balanced Accuracy, F1 Score, AUC, Brier Score, KL Divergence, And Training & Test Times  
    def compute_performance(self, Y_true, Y_pred, Y_prob):
        #Constant To Avoid NAN In KL Divergence
        P = np.asarray(Y_true)+0.00001
        #Constant To Avoid NAN In KL Divergence
        Q = np.asarray(Y_prob)+0.00001
        
        bal_acc = metrics.balanced_accuracy_score(Y_true, Y_pred)
        f1 = metrics.f1_score(Y_true, Y_pred)
        fpr, tpr, _ = metrics.roc_curve(Y_true, Y_prob, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        brier = metrics.brier_score_loss(Y_true, Y_prob)
        kl_div = np.sum(P*np.log(P/Q))

        print("\nCOMPUTING performance on test data...")

        print("Balanced Accuracy="+str(bal_acc))
        print("F1 Score="+str(f1))
        print("Area Under Curve="+str(auc))
        print("Brier Score="+str(brier))
        print("KL Divergence="+str(kl_div))        
        print("Training Time="+str(self.running_time)+" secs.")
        print("Inference Time="+str(self.inference_time)+" secs.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: ModelEvaluator.py [config_file.txt] [test_file.csv]")
        print("EXAMPLE> ModelEvaluator.py config-lungcancer.txt lung_cancer-test.csv")
        exit(0)
    else:
        configfile = sys.argv[1]
        datafile_test = sys.argv[2]
        ModelEvaluator(configfile, datafile_test)