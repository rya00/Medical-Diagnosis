import bnlearn as bn
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Set Training Data, Network Structure, And Type Of Conditional Independence Test
TRAINING_DATA = 'data/continuous/parkinson_data-VOICE-features-train1.csv'
CONDITIONAL_INDEPENDENCE_TEST = 'cressie_read'

# Data Loading Using Pandas
data = pd.read_csv(TRAINING_DATA, encoding='UTF-8')
print("DATA:\n", data)

# Definition Of Directed Acyclic Graphs (Predefined Structures)
edges = [('status', 'MDVP_Fo_Hz'), ('status', 'MDVP_Fhi_Hz'), ('status', 'MDVP_Flo_Hz'), ('status', 'MDVP_Jitter_%'), ('status', 'MDVP_Jitter_Abs'), ('status', 'MDVP_RAP'), ('status', 'MDVP_PPQ'), ('status', 'Jitter_DDP'), ('status', 'MDVP_Shimmer'), ('status', 'MDVP_Shimmer_dB'), ('status', 'Shimmer_APQ3'), ('status', 'Shimmer_APQ5'), ('status', 'MDVP_APQ'), ('status', 'Shimmer_DDA'), ('status', 'NHR'), ('status', 'HNR'), ('status', 'RPDE'), ('status', 'DFA'), ('status', 'spread1'), ('status', 'spread2'), ('status', 'D2'), ('status', 'PPE')]

# Creation Of Directed Acyclic Graph (DAG)
DAG = bn.make_DAG(edges)
print("DAG:\n", DAG)

# Parameter Learning Using Maximum Likelihood Estimation
model = bn.parameter_learning.fit(DAG, data, methodtype="maximumlikelihood")
print("model=", model)

# Statistical Test Of Independence
if model is not None:
    model = bn.independence_test(model, data, test=CONDITIONAL_INDEPENDENCE_TEST, alpha=0.05)
    ci_results = list(model['independence_test']['stat_test'])
    num_edges2remove = ci_results.count(False)
    print(model['independence_test'])
    print("num_edges2remove = " + str(num_edges2remove))

    # Visualize the learnt structure
    G = nx.DiGraph()
    G.add_edges_from(model['model_edges'])
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
    plt.title('Directed Acyclic Graph (DAG)')
    plt.show()
else:
    print("Model learning failed.")