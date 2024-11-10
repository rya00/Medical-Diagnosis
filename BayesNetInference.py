#############################################################################
# BayesNetInference.py
#
# This program implements the algorithm "Inference by Enumeration", which
# makes use of BayesNetsReader to facilitate reading data of a Bayes net via
# the object self.bn created by the inherited class (BayesNetReader). It also
# makes use of miscellaneous methods implemented in BayesNetUtil.
# Its purpose is to answer probabilistic queries such as P(Y|X=true,Z=false).
# This implementation is agnostic of the data and provides a general
# implementation that can ne used across datasets by providing a config file.
#
# This program also implements the algorithm "Rejection Sampling", which
# imports functionalities to facilitate reading data of a Bayes net via
# the object self.bn created by the inherited class BayesNetReader.
# Its purpose is to answer probabilistic queries such as P(Y|X=true,Z=false).
# This implementation is agnostic of the data and provides a general
# implementation that can be used across datasets by providing a config file.
#
# Whilst Inference by Enumeration has been tested with discrete and Gaussian
# Bayes nets, Rejection Sampling has only been tested with discrete Bayes nets.
#
# WARNINGS:
#    (1) This code has been revised with multiple cases but not thoroughly tested.
#    (2) The execution time depends on the number of random samples.
#
# Version: 1.0, Date: 06 October 2022, 1st version of class BayesNetExactInference
# Version: 1.1, Date: 07 October 2022, 1st version of class BayesNetApproxInference
# Version: 1.2, Date: 21 October 2022, revised code to be more query compatible
# Version: 1.3, Date: 20 October 2022, revised code made simpler (more intuitive)
# Version: 1.4, Date: 13 October 2023, merged classes for unified prob. inference
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import random
import time
import BayesNetUtil as bnu
from BayesNetReader import BayesNetReader


class BayesNetInference(BayesNetReader):
    query = {}
    prob_dist = {}
    verbose = False

    def __init__(self, alg_name, file_name, prob_query, num_samples):
        super().__init__(file_name)

        if alg_name is None and prob_query is None:
            return

        self.query = bnu.tokenise_query(prob_query, self.verbose)

        start = time.time()
        if alg_name == 'InferenceByEnumeration':
            self.prob_dist = self.enumeration_ask()
            self.prob_dist = self.enumeration_ask()
            normalised_dist = bnu.normalise(self.prob_dist)
            print("unnormalised P(%s)=%s" % (self.query["query_var"], self.prob_dist))
            print("normalised P(%s)=%s" % (self.query["query_var"], normalised_dist))

        elif alg_name == 'RejectionSampling':
            self.prob_dist = self.rejection_sampling(num_samples)
            print("P(%s)=%s" %(self.query["query_var"], self.prob_dist))

        else:
            print("ERROR: Couldn't recognise algorithm="+str(alg_name))
            print("Valid choices={InferenceByEnumeration,RejectionSampling}")

        end = time.time()
        print('Execution Time: {}'.format(end-start))

    # main method for inference by enumeration, which invokes
	# enumerate_all() for each domain value of the query variable
    def enumeration_ask(self):
        if self.verbose: print("\nSTARTING Inference by Enumeration...")

        if "regression_models" not in self.bn:
            # initialisation required for discrete Bayes nets
            Q = {}
            for value in self.bn["rv_key_values"][self.query["query_var"]]:
                value = value.split('|')[0]
                Q[value] = 0
        else:
            # initialisation required for Gaussian Bayes nets
            Q = {0.0: 0, 1.0: 0}

        for value, probability in Q.items():
            variables = self.bn["random_variables"].copy()
            evidence = self.query["evidence"].copy()
            evidence[self.query["query_var"]] = value
            probability = self.enumerate_all(variables, evidence)
            Q[value] = probability

        if self.verbose: print("\tQ="+str(Q))
        return Q # Q is an unnormalised probability distribution

    # returns a probability for the arguments provided, based on
	# summations or multiplications of prior/conditional probabilities
    def enumerate_all(self, variables, evidence):
        if len(variables) == 0:
            return 1.0

        V = variables[0]

        if V in evidence:
            v = str(evidence[V]).split('|')[0]
            p = bnu.get_probability_given_parents(V, v, evidence, self.bn)
            variables.pop(0)
            return p*self.enumerate_all(variables, evidence)

        else:
            sum = 0
            evidence_copy = evidence.copy()
            for v in bnu.get_domain_values(V, self.bn):
                evidence[V] = v
                p = bnu.get_probability_given_parents(V, v, evidence, self.bn)
                rest_variables = variables.copy()
                rest_variables.pop(0)
                sum += p*self.enumerate_all(rest_variables, evidence)
                evidence = evidence_copy

            return sum

    # main method to carry out approximate probabilistic inference,
	# which invokes prior_sample() and is_compatible_with_evidence()
    def rejection_sampling(self, num_samples):
        query_variable = self.query["query_var"]
        evidence = self.query["evidence"]
        samples = [] # vector of non-rejected samples
        C = {} # counts per value in query variable

        print("\nSTARTING rejection sampling...")
        print("query_variable="+str(query_variable))
        print("evidence="+str(evidence))

        # initialise vector of counts
        for value in self.bn["rv_key_values"][query_variable]:
            value = value.split("|")[0]
            C[value] = 0

        # loop to increase counts when the sampled vector X consistent w/evidence
        for i in range(0, num_samples):
            X = self.prior_sample(evidence)
            if X != None and self.is_compatible_with_evidence(X, evidence):
                value_to_increase = X[query_variable]
                C[value_to_increase] += 1

        try:
            print("Countings of query_variable %s=%s" % (query_variable, C))
            return bnu.normalise(C)
        except:
            print("ABORTED due to insufficient number of samples...")
            exit(0)

    # returns a dictionary of sampled values for each of the random variables
    def prior_sample(self, evidence):
        X = {}
        sampled_var_values = {}

        # iterates over the set of random variables as specified in the 
		# config file of the Bayes net, in the order from left to right
        for variable in self.bn["random_variables"]:
            X[variable] = self.get_sampled_value(variable, sampled_var_values)
            sampled_var_values[variable] = X[variable]
            if variable in evidence and evidence[variable] != X[variable]:
                if self.verbose: 
                    print("RETURNING X=",X," var=",variable," in e=",evidence)
                return None

        return X

    # returns a sampled value for the given random variable as argument
    def get_sampled_value(self, V, sampled):
        parents = bnu.get_parents(V, self.bn)
        cumulative_cpt = {}
        prob_mass = 0

        # generate cummulative distribution for random variable V without parents
        if parents is None:
            for value, probability in self.bn["CPT("+V+")"].items():
                prob_mass += probability
                cumulative_cpt[value] = prob_mass

        # generate cummulative distribution for random variable V with parents
        else:
            for v in bnu.get_domain_values(V, self.bn):
                p = bnu.get_probability_given_parents(V, v, sampled, self.bn)
                prob_mass += p
                cumulative_cpt[v] = prob_mass

        # check that the probabilities sum to 1 (or almost)
        if prob_mass < 0.999 or prob_mass > 1.001:
            print("ERROR: probabilities=%s do not sum to 1" % (cumulative_cpt))
            exit(0)

		# sample a value from the cummulative distribution generated above
        for value, probability in cumulative_cpt.items():
            random_number = random.random()
            if random_number <= probability:
                return value.split("|")[0]

        return None # this shouldn't happen -- unless something was incompatible

    # returns True if evidence has key-value pairs same as X
	# returns False otherwise
    def is_compatible_with_evidence(self, X, evidence):
        compatible = True
        print("X=%s" % (X))
        for variable, value in evidence.items():
            if self.verbose: 
                print("*variable=%s value=%s" % (variable, value))
            #print("X[variable]=",X[variable])
            #if variable not in X or X[variable] != value:
            if X[variable] != value:
                compatible = False
                break
        return compatible


if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("USAGE: BayesNetInference.py [inference_algorithm] [your_config_file.txt] [query] (num_samples)")
        print("EXAMPLE1> BayesNetInference.py InferenceByEnumeration config-alarm.txt \"P(B|J=true,M=true)\"")
        print("EXAMPLE2> BayesNetInference.py RejectionSampling config-alarm.txt \"P(B|J=true,M=true)\" 10000")
        exit(0)

    alg_name = sys.argv[1] # inference algorithm={InferenceByEnumeration,RejectionSampling}
    file_name = sys.argv[2] # your_config_file.txt, e.g., config-alarm.txt
    prob_query = sys.argv[3] # query, e.g., P(B|J=true,M=true)
    num_samples = int(sys.argv[4]) if len(sys.argv)==5 else None # number of samples, e.g., 10000

    BayesNetInference(alg_name, file_name, prob_query, num_samples)
