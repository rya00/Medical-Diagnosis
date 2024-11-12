import sys
import time
from BayesNetReader import BayesNetReader
from DataReader import CSV_DataReader

#Declare CPT Generator Class
class CPT_Generator(BayesNetReader):
    configfile_name = None
    bn = None
    nbc = None
    countings = {}
    CPTs = {}
    #Constant_L Set To One To Avoid Zero Probabilities
    constant_l = 1  

    def __init__(self, configfile_name, datafile_name):
        self.configfile_name = configfile_name
        self.bn = BayesNetReader(configfile_name)
        self.csv = CSV_DataReader(datafile_name)
        self.running_time = time.time()
        self.generate_prior_and_conditional_countings()
        self.generate_probabilities_from_countings()
        self.write_CPTs_to_configuration_file()
        self.running_time = time.time() - self.running_time
        print("Training Time="+str(self.running_time)+" secs.")

    def generate_prior_and_conditional_countings(self):
        print("\nGENERATING countings for prior/conditional distributions...")
        print("-------------------------------------------------------------")

        for pd in self.bn.bn["structure"]:
            print(str(pd))
            p = pd.replace('(', ' ').replace('\ufeff', '')
            p = p.replace(')', ' ').replace('\ufeff', '')
            tokens = p.split("|")

            #Generate Countings For Prior Probabilities
            if len(tokens) == 1:
                variable = tokens[0].split(' ')[1].replace('\ufeff', '')
                variable_index = self.get_variable_index(variable)
                counts = self.initialise_counts(variable)
                self.get_counts(variable_index, None, counts)

            #Generate Countings For Conditional Probabilities
            if len(tokens) == 2:
                variable = tokens[0].split(' ')[1].replace('\ufeff', '')
                variable_index = self.get_variable_index(variable)
                parents = tokens[1].strip().split(',')
                parent_indexes = self.get_parent_indexes(parents)
                counts = self.initialise_counts(variable, parents)
                self.get_counts(variable_index, parent_indexes, counts)

            self.countings[pd] = counts
            print("counts="+str(counts))
            print()

    def generate_probabilities_from_countings(self):
        print("\nGENERATING prior and conditional probabilities...")
        print("---------------------------------------------------")

        for pd, counts in self.countings.items():
            print(str(pd))
            tokens = pd.split("|")
            print(f"tokens={tokens}")

            variable = tokens[0].replace("P(", "").replace('\ufeff', '')
            cpt = {}

            #Generate Prior Probabilities
            if len(tokens) == 1:
                _sum = 0
                for key, count in counts.items():
                    _sum += count

                Jl = len(counts)*self.constant_l
                for key, count in counts.items():
                    cpt[key] = (count+self.constant_l)/(_sum+Jl)

            #Generate Conditional Probabilities
            if len(tokens) == 2:
                parents_values = self.get_parent_values(counts)
                for parents_value in parents_values:
                    _sum = 0
                    for key, count in counts.items():
                        if key.endswith("|"+parents_value):
                            _sum += count

                    J = len(self.csv.rv_key_values[variable])
                    Jl = J*self.constant_l
                    for key, count in counts.items():
                        if key.endswith("|"+parents_value):
                            cpt[key] = (count+self.constant_l)/(_sum+Jl)

            self.CPTs[pd] = cpt
            print("CPT="+str(cpt))
            print()

    def get_variable_index(self, variable):
        for i in range(0, len(self.csv.rand_vars)):
            if variable == self.csv.rand_vars[i]:
                return i
        print("WARNING: couldn't find index of variables=%s" % (variable))
        return None

    def get_parent_indexes(self, parents):
        indexes = []
        for parent in parents:
            index = self.get_variable_index(parent)
            indexes.append(index)
        return indexes

    def get_parent_values(self, counts):
        values = []
        for key, count in counts.items():
            value = key.split('|')[1]
            if value not in values:
                values.append(value)
        return values

    def initialise_counts(self, variable, parents=None):
        counts = {}

        if parents is None:
            #Initialise Counts Of Variables Without Parents
            for var_val in self.csv.rv_key_values[variable]:
                if var_val not in counts:
                    counts[var_val] = 0

        else:
            #Enumerate All Sequence Values Of Parent Variables
            parents_values = []
            last_parents_values = []
            for i in range(0, len(parents)):
                parent = parents[i]
                for var_val in self.csv.rv_key_values[parent]:
                    if i == 0:
                        parents_values.append(var_val)
                    else:
                        for last_val in last_parents_values:
                            parents_values.append(last_val+','+var_val)

                last_parents_values = parents_values.copy()
                parents_values = []

            #Initialise Counts Of Variables With Parents
            for var_val in self.csv.rv_key_values[variable]:
                for par_val in last_parents_values:
                    counts[var_val+'|'+par_val] = 0

        return counts

    def get_counts(self, variable_index, parent_indexes, counts):
        #Accumulate Countings
        for values in self.csv.rv_all_values:
            if parent_indexes is None:
                #Case Prior Probability
                value = values[variable_index]
            else:
                #Case Conditional Probability
                parents_values = ""
                for parent_index in parent_indexes:
                    value = values[parent_index]
                    if len(parents_values) == 0:
                        parents_values = value
                    else:
                        parents_values += ','+value
                value = values[variable_index]+'|'+parents_values
            counts[value] += 1

    def write_CPTs_to_configuration_file(self):
        print("\nWRITING config file with CPT tables...")
        print("See rewritten file "+str(self.configfile_name))
        print("---------------------------------------------------")
        name = self.bn.bn["name"]

        rand_vars = self.bn.bn["random_variables_raw"]
        rand_vars = str(rand_vars).replace('[', '').replace(']', '').replace('\ufeff', '')
        rand_vars = str(rand_vars).replace('\'', '').replace(', ', ';').replace('\ufeff', '')

        structure = self.bn.bn["structure"]
        structure = str(structure).replace('[', '').replace(']', '').replace('\ufeff', '')
        structure = str(structure).replace('\'', '').replace(', ', ';').replace('\ufeff', '')

        with open(self.configfile_name, 'w', encoding='UTF-8') as cfg_file:
            cfg_file.write("name:"+str(name))
            cfg_file.write('\n')
            cfg_file.write('\n')
            cfg_file.write("random_variables:"+str(rand_vars))
            cfg_file.write('\n')
            cfg_file.write('\n')
            cfg_file.write("structure:"+str(structure))
            cfg_file.write('\n')
            cfg_file.write('\n')
            for key, cpt in self.CPTs.items():
                cpt_header = key.replace("P(", "CPT(").replace('\ufeff', '')
                cfg_file.write(str(cpt_header)+":")
                cfg_file.write('\n')
                num_written_probs = 0
                for domain_vals, probability in cpt.items():
                    num_written_probs += 1
                    line = str(domain_vals)+"="+str(probability)
                    line = line+";" if num_written_probs < len(cpt) else line
                    cfg_file.write(line)
                    cfg_file.write('\n')
                cfg_file.write('\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: CPT_Generator.py [your_config_file.txt] [training_file.csv]")
        print("EXAMPLE> CPT_Generator.py config-playtennis.txt play_tennis-train.csv")
        exit(0)
    else:
        configfile_name = sys.argv[1]
        datafile_name = sys.argv[2]
        CPT_Generator(configfile_name, datafile_name)