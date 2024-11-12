import bnlearn as bn
import pandas as pd

# definition of directed acyclic graphs (predefined structures)
edges = [('status', 'MDVP_Fo_Hz'), ('status', 'MDVP_Fhi_Hz'), ('status', 'MDVP_Flo_Hz'), ('status', 'MDVP_Jitter_%'), ('status', 'MDVP_Jitter_Abs'), ('status', 'MDVP_RAP'), ('status', 'MDVP_PPQ'), ('status', 'Jitter_DDP'), ('status', 'MDVP_Shimmer'), ('status', 'MDVP_Shimmer_dB'), ('status', 'Shimmer_APQ3'), ('status', 'Shimmer_APQ5'), ('status', 'MDVP_APQ'), ('status', 'Shimmer_DDA'), ('status', 'NHR'), ('status', 'HNR'), ('status', 'RPDE'), ('status', 'DFA'), ('status', 'spread1'), ('status', 'spread2'), ('status', 'D2'), ('status', 'PPE')]

# examples of training data include 'data\lung_cancer-train.csv' or  'data\lang_detect_train.csv', etc.
# examples of net structure (as below): edges_langdet1, edges_langdet2, edges, edges_lungcancer2
# choices of CI test: chi_square, g_sq, log_likelihood, freeman_tuckey, modified_log_likelihood, neyman, cressie_read
TRAINING_DATA = 'data/continuous/parkinson_data-VOICE-features-train1.csv'
NETWORK_STRUCTURE = edges 
CONDITIONAL_INDEPENDENCE_TEST = 'cressie_read'

# data loading using pandas
data = pd.read_csv(TRAINING_DATA, encoding='latin')
print("DATA:\n", data)

 # creation of the directed acyclic graph (DAG)
DAG = bn.make_DAG(NETWORK_STRUCTURE)
print("DAG:\n", DAG)

# parameter learning using Maximum Likelihood Estimation
model = bn.parameter_learning.fit(DAG, data, methodtype="maximumlikelihood")
print("model=",model)

# statististical test of independence
model = bn.independence_test(model, data, test=CONDITIONAL_INDEPENDENCE_TEST, alpha=0.05)
ci_results = list(model['independence_test']['stat_test'])
num_edges2remove = ci_results.count(False)
print(model['independence_test'])
print("num_edges2remove="+str(num_edges2remove))
