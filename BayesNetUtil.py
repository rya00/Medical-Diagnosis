#############################################################################
# BayesNetUtil.py
#
# Implements functions to simplify the implementation of algorithms for
# probabilistic inference with Bayesian networks.
#
# Version: 1.0, 06 October 2022
# Version: 1.1, 08 October 2023 commented and extended to support loop detection
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import numpy as np
import networkx as nx


# example query: P(B|J=true,M=true)
# example dictionary: query={query_var:'B', evidence:'J=true,M=true'}
# example tokenised dictionary: query={query_var:'B', evidence:{'J':'true','M':'true'}}
# returns a tokenised dictionary as in the example above
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


# returns the parent of random variable 'child' given Bayes Net 'bn'
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

# returns the probability of tuple V=v (where V is a random variable and
# v is a domain value) given the evidence and Bayes Net (bn) provided
def get_probability_given_parents(V, v, evidence, bn):
    parents = get_parents(V, bn)
    is_gaussian = True if "regression_models" in bn else False
    probability = 0

    if parents is None and is_gaussian == False:
        cpt = bn["CPT("+V+")"]
        probability = cpt[v]

    elif parents is not None and is_gaussian == False:
        cpt = bn["CPT("+V+"|"+parents+")"]
        values = v
        for parent in parents.split(","):
            separator = "|" if values == v else ","
            values = values + separator + evidence[parent]
        probability = cpt[values]

    elif parents is None and is_gaussian == True:
        mean = bn["means"][V]
        std = bn["stdevs"][V]
        probability = get_gaussian_density(float(v), mean, std)
        #print("V=%s v=%s mean=%s std=%s pd=%s p=%s" % (V, v, mean, std, prob_density, probability))

    elif parents is not None and is_gaussian == True:
        values = []
        parent_list = parents.split(",")
        for i in range(0, len(parent_list)):
            values.append(float(evidence[parent_list[i]]))
        values = np.asarray([values])
        regressor = bn["regressors"][V]
        pred_mean = regressor.predict(values)
        std = bn["stdevs"][V]
        probability = get_gaussian_density(float(v), pred_mean, std)
        #print("V=%s v=%s mean=%s std=%s pd1=%s pd2=%s p=%s" % (V, v, pred_mean, pred_std, probA, probB, probability))

    else:
        print("ERROR: Don't know how to get probability for V="+str(V))
        exit(0)

    return probability


# returns the domain values of random variable 'V' given Bayes Net 'bn'
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


# returns the number of probabilities (full enumeration) of random variable 'V',
# which is currently used to calculate the penalty of the BIC scoring function.
def get_number_of_probabilities(V, bn):
    for key, cpt in bn.items():
        if key == "CPT("+V+")":
            return len(cpt.keys())

        elif key.startswith("CPT("+V+"|"):
            return len(cpt.items())


# returns the index of random variable 'V' given Bayes Net 'bn'
def get_index_of_variable(V, bn):
    for i in range(0, len(bn["random_variables"])):
        variable = bn["random_variables"][i]
        if V == variable:
            return i

    print("ERROR: Couldn't find index of variable "+str(V))
    exit(0)


# returns a normalised probability distribution of the provided counts,
# where counts is a dictionary of domain_value-counts
def normalise(counts):
    _sum = 0
    for value, count in counts.items():
        _sum += count

    distribution = {}
    for value, count in counts.items():
        if _sum == 0: p = 0.5 # default if _sum=0
        else: p = float(count/_sum)
        distribution[value] = p

    return distribution


# requires the following dependency: pip install networkx
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

# returns the probability density of the given arguments
def get_gaussian_density(x, mean, stdev):
    e_val = -0.5*np.power((x-mean)/stdev, 2)
    probability = (1/(stdev*np.sqrt(2*np.pi))) * np.exp(e_val)
    return probability