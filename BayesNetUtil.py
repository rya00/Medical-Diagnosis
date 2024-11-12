import numpy as np
import networkx as nx

#Returns Tokenised Dictionary
def tokenise_query(prob_query, verbose):
    if verbose: print("\nTOKENISING probabilistic query="+str(prob_query))

    query = {}
    prob_query = prob_query[2:]
    prob_query = prob_query[:len(prob_query)-1]
    query["query_var"] = prob_query.split("|")[0]
    query["evidence"] = prob_query.split("|")[1]

    evidence = {}
    if query["evidence"].find(','):
        for pair in query["evidence"].split(','):
            tokens = pair.split('=')
            evidence[tokens[0]] = tokens[1]
        query["evidence"] = evidence

    if verbose: print("query="+str(query))
    return query

#Returns Parent Of Random Variable 'Child' Given Bayes Net 'bn'
def get_parents(child, bn):
    for conditional in bn["structure"]:
        if conditional.startswith("P("+child+")"):
            return None
        elif conditional.startswith("P("+child+"|"):
            parents = conditional.split("|")[1]
            parents = parents[:len(parents)-1]
            return parents

    print("ERROR: Couldn't find parent(s) of variable "+str(child))
    exit(0)

def get_probability_given_parents(V, v, evidence, bn):
    parents = get_parents(V, bn)
    is_gaussian = "regression_models" in bn
    probability = 0

    if parents is None and not is_gaussian:
        cpt = bn["CPT("+V+")"]
        probability = cpt.get(v, 0)  # Default to 0 if not found

    elif parents is not None and not is_gaussian:
        cpt = bn["CPT("+V+"|"+parents+")"]
        values = v
        
        # Constructing the values string
        for parent in parents.split(","):
            separator = "|" if values == v else ","
            values += separator + evidence[parent].strip(') ')  # Strip unwanted characters
        
        # Try to find an exact match first
        probability = cpt.get(values, 0)  # Default to 0 if not found

        if probability == 0:  # If no exact match, try finding the closest match
            closest_key = find_closest_key(values, cpt.keys())
            probability = cpt.get(closest_key, 0)  # Default to 0 if closest key is also not found

    elif parents is None and is_gaussian:
        mean = bn["means"][V]
        std = bn["stdevs"][V]
        probability = get_gaussian_density(float(v), mean, std)

    elif parents is not None and is_gaussian:
        values = [float(evidence[parent]) for parent in parents.split(",")]
        regressor = bn["regressors"][V]
        pred_mean = regressor.predict(np.array([values]))
        std = bn["stdevs"][V]
        probability = get_gaussian_density(float(v), pred_mean, std)

    else:
        print("ERROR: Don't know how to get probability for V="+str(V))
        exit(0)

    return probability

def find_closest_key(target, keys):
    def parse_values(value_string):
        return [float(x) for x in value_string.replace(',', '|').split('|') if x]

    target_values = parse_values(target)
    min_diff = float('inf')
    closest_key = None

    for key in keys:
        key_values = parse_values(key)
        if len(key_values) != len(target_values):
            continue  # Skip keys with different number of values
        diff = sum(abs(t - k) for t, k in zip(target_values, key_values))
        if diff < min_diff:
            min_diff = diff
            closest_key = key

    return closest_key

#Returns Domain Values Of Random Variable 'V' Given Bayes Net 'bn'
def get_domain_values(V, bn):
    domain_values = []

    for key, cpt in bn.items():
        if key == "CPT("+V+")":
            domain_values = list(cpt.keys())

        elif key.startswith("CPT("+V+"|"):
            for entry, prob in cpt.items():
                value = entry.split("|")[0]
                if value not in domain_values:
                    domain_values.append(value)

    if len(domain_values) == 0:
        print("ERROR: Couldn't find values of variable "+str(V))
        exit(0)

    return domain_values

#Returns Number Of Probabilities (Full Enumeration) Of Random Variable 'V', Which Is Currently Used To Calculate Penalty Of BIC Scoring Function
def get_number_of_probabilities(V, bn):
    for key, cpt in bn.items():
        if key == "CPT("+V+")":
            return len(cpt.keys())

        elif key.startswith("CPT("+V+"|"):
            return len(cpt.items())

#Returns Index Of Random Variable 'V' Given Bayes Net 'bn'
def get_index_of_variable(V, bn):
    for i in range(0, len(bn["random_variables"])):
        variable = bn["random_variables"][i]
        if V == variable:
            return i

    print("ERROR: Couldn't find index of variable "+str(V))
    exit(0)

#Returns Normalised Probability Distribution Of Provided Counts, Where Counts Is Dictionary Of Domain_Value-Counts
def normalise(counts):
    _sum = 0
    for value, count in counts.items():
        _sum += count

    distribution = {}
    for value, count in counts.items():
        if _sum == 0: p = 0.5
        else: p = float(count/_sum)
        distribution[value] = p

    return distribution

def has_cycles(edges):
    print("\nDETECTING cycles in graph %s" % (edges))
    G = nx.DiGraph(edges)

    cycles = False
    for cycle in nx.simple_cycles(G):
        print("Cycle found:"+str(cycle))
        cycles = True

    if cycles is False:
        print("No cycles found!")
    return cycles

#Returns Probability Density Of Given Arguments
def get_gaussian_density(x, mean, stdev):
    e_val = -0.5*np.power((x-mean)/stdev, 2)
    probability = (1/(stdev*np.sqrt(2*np.pi))) * np.exp(e_val)
    return probability