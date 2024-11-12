import sys
import random
import time
import BayesNetUtil as bnu
from BayesNetReader import BayesNetReader

#Declare Bayes Net Inference Class
class BayesNetInference(BayesNetReader):
    query = {}
    prob_dist = {}
    verbose = False

    def __init__(self, alg_name, file_name, prob_query, num_samples):
        super().__init__(file_name)

        if alg_name is None and prob_query is None:
            return

        self.query = bnu.tokenise_query(prob_query, self.verbose)
        self.preprocess_query()  # Add this line to preprocess the query

        start = time.time()
        if alg_name == 'InferenceByEnumeration':
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

    def preprocess_query(self):
    # Check if the query is for Parkinson's dataset
        if "MDVP" in self.query['query_var']:  # This checks if it's a Parkinson's query
            for key, value in self.query['evidence'].items():
                value = value.strip(') ')  # Remove trailing parenthesis and spaces
                try:
                    if key == 'MDVP_Jitter_Abs':
                        self.query['evidence'][key] = '{:.8f}'.format(float(value))
                    elif key in ['MDVP_Fo_Hz', 'MDVP_Fhi_Hz', 'MDVP_Flo_Hz', 'MDVP_Shimmer_dB', 'HNR']:
                        self.query['evidence'][key] = '{:.3f}'.format(float(value))
                    else:
                        self.query['evidence'][key] = '{:.5f}'.format(float(value))
                except ValueError as e:
                    print(f"Error converting value for {key}: {value}")
                    raise e  # Re-raise the error after logging it
        else:
            # For dementia queries, do nothing or apply different logic if needed
            print("No preprocessing applied for dementia queries.")

    #Method For Inference By Enumeration, Which Invokes Enumerate_All() For Each Domain Value Of Query Variable
    def enumeration_ask(self):
        if self.verbose: print("\nSTARTING Inference by Enumeration...")

        if "regression_models" not in self.bn:
            #Q Is An Unnormalised Probability Distribution
            Q = {}
            for value in self.bn["rv_key_values"][self.query["query_var"]]:
                value = value.split('|')[0]
                Q[value] = 0
        else:
            Q = {0.0: 0, 1.0: 0}

        for value, probability in Q.items():
            variables = self.bn["random_variables"].copy()
            evidence = self.query["evidence"].copy()
            evidence[self.query["query_var"]] = value
            probability = self.enumerate_all(variables, evidence)
            Q[value] = probability

        if self.verbose: print("\tQ="+str(Q))
        return Q

    #Returns Probability For Arguments Provided, Based On Summations Or Multiplications Of Prior/Conditional Probabilities
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

    #Method To Carry Out Approximate Probabilistic Inference Which Invokes Prior_Sample() And Is_Compatible_With_Evidence()
    def rejection_sampling(self, num_samples):
        query_variable = self.query["query_var"]
        evidence = self.query["evidence"]
        #Vector Of Non Rejected Samples
        samples = []
        #Counts Per Value In Query Variable
        C = {} 

        print("\nSTARTING rejection sampling...")
        print("query_variable="+str(query_variable))
        print("evidence="+str(evidence))

        #Initialise Vector Of Counts
        for value in self.bn["rv_key_values"][query_variable]:
            value = value.split("|")[0]
            C[value] = 0

        #Loop To Increase Counts When Sampled Vector X Consistent With Evidence
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

    #Returns Dictionary Of Sampled Values For Each Of Random Variables
    def prior_sample(self, evidence):
        X = {}
        sampled_var_values = {}

        #Iterates Over Set Of Random Variables As Specified In Order From Left To Right
        for variable in self.bn["random_variables"]:
            X[variable] = self.get_sampled_value(variable, sampled_var_values)
            sampled_var_values[variable] = X[variable]
            if variable in evidence and evidence[variable] != X[variable]:
                if self.verbose: 
                    print("RETURNING X=",X," var=",variable," in e=",evidence)
                return None

        return X

    #Returns Sampled Value For Given Random Variable As Argument
    def get_sampled_value(self, V, sampled):
        parents = bnu.get_parents(V, self.bn)
        cumulative_cpt = {}
        prob_mass = 0

        #Generate Cumulative Distribution For Random Variable V Without Parents
        if parents is None:
            for value, probability in self.bn["CPT("+V+")"].items():
                prob_mass += probability
                cumulative_cpt[value] = prob_mass

        #Generate Cumulative Distribution For Random Variable V With Parents
        else:
            for v in bnu.get_domain_values(V, self.bn):
                p = bnu.get_probability_given_parents(V, v, sampled, self.bn)
                prob_mass += p
                cumulative_cpt[v] = prob_mass

        #Check That Probabilities Sum To One (or almost)
        if prob_mass < 0.999 or prob_mass > 1.001:
            print("ERROR: probabilities=%s do not sum to 1" % (cumulative_cpt))
            exit(0)

		#Sample Value From Above Generated Cumulative Distribution
        for value, probability in cumulative_cpt.items():
            random_number = random.random()
            if random_number <= probability:
                return value.split("|")[0]

        return None

    #Returns True If Evidence Has Key-Value Pairs Same As X Otherwise Returns False 
    def is_compatible_with_evidence(self, X, evidence):
        compatible = True
        print("X=%s" % (X))
        for variable, value in evidence.items():
            if self.verbose: 
                print("*variable=%s value=%s" % (variable, value))
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

    alg_name = sys.argv[1]
    file_name = sys.argv[2]
    prob_query = sys.argv[3]
    num_samples = int(sys.argv[4]) if len(sys.argv) == 5 else None

    BayesNetInference(alg_name, file_name, prob_query, num_samples)