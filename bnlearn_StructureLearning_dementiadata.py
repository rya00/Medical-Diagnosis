import bnlearn as bn
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Set Training Data, Network Structure, And Type Of Conditional Independence Test
TRAINING_DATA = 'data/continuous/dementia_data-MRI-features-train1.csv'
CONDITIONAL_INDEPENDENCE_TEST = 'cressie_read'

# Data Loading Using Pandas
data = pd.read_csv(TRAINING_DATA, encoding='UTF-8')
print("DATA:\n", data)

# Definition Of Directed Acyclic Graphs (Predefined Structures)
edges = [
    ('Age', 'eTIV'),          # eTIV depends on Age
    ('eTIV', 'ASF'),          # ASF depends on eTIV
    ('eTIV', 'nWBV'),         # nWBV depends on eTIV
    ('Age', 'MMSE'),          # MMSE depends on Age
    ('EDUC', 'MMSE'),         # MMSE also depends on EDUC
    ('MMSE', 'CDR'),          # CDR depends on MMSE
    ('nWBV', 'Group'),        # Group classification may depend on nWBV
    ('MMSE', 'Group'),        # Group classification may also depend on MMSE
    ('CDR', 'Group')          # Group classification may also depend on CDR
]

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