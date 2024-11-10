import sys

#Declare Bayes Net Reader Class
class BayesNetReader:
    def __init__(self, file_name):
        #Make bn An Instance Variable
        self.bn = {}
        self.read_data(file_name)
        self.tokenise_data()

    #Starts Loading Configuration File Into Dictionary 'bn', By Splitting Strings With Character ':' And Storing Keys & Values 
    def read_data(self, data_file):
        print("\nREADING data file %s..." % (data_file))

        try:
            with open(data_file, encoding='utf-8-sig') as cfg_file:
                key = None
                value = None
                for line in cfg_file:
                    line = line.strip().replace('\ufeff', '')
                    if len(line) == 0:
                        continue

                    tokens = line.split(":")
                    if len(tokens) == 2:
                        if value is not None:
                            self.bn[key] = value
                            value = None

                        key = tokens[0].replace('\ufeff', '')
                        value = tokens[1].replace('\ufeff', '')
                    else:
                        value += tokens[0].replace('\ufeff', '')

                #Ensure Last Key-Value Pair Is Added
                if key and value is not None:  
                    self.bn[key] = value
                self.bn["random_variables_raw"] = self.bn.get("random_variables", "")
                print("RAW key-values=" + str(self.bn))
        except FileNotFoundError:
            print(f"Error: The file '{data_file}' was not found.")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred: {e}")
            sys.exit(1)

    #Continues Loading Configuration File Into Dictionary 'bn', By Separating Key-Value Pairs
    def tokenise_data(self):
        print("TOKENISING data...")
        rv_key_values = {}

        for key, values in self.bn.items():
            if key == "random_variables":
                var_set = []
                for value in values.split(";"):
                    if value.find("(") > -1 and value.find(")") > -1:
                        value = value.replace('(', ' ').replace(')', ' ').replace('\ufeff', '')
                        parts = value.split(' ')
                        var_set.append(parts[1].strip())
                    else:
                        var_set.append(value)
                self.bn[key] = var_set

            elif key.startswith("CPT"):
                #Store Conditional Probability Tables (CPTs) As Dictionaries
                cpt = {}
                total_prob = 0
                for value in values.split(";"):
                    pair = value.split("=")
                    #Check For Valid Key-Value Pairs
                    if len(pair) == 2:  
                        cpt[pair[0].replace('\ufeff', '')] = float(pair[1].replace('\ufeff', ''))
                        total_prob += float(pair[1].replace('\ufeff', ''))
                print("key=%s cpt=%s total_prob=%s" % (key, cpt, total_prob))
                self.bn[key] = cpt

                #Store Unique Values For Each Random Variable
                rand_var = key[4:].split("|")[0] if "|" in key else key[4:].split(")")[0]
                unique_values = list(cpt.keys())
                rv_key_values[rand_var.replace('\ufeff', '')] = unique_values

            else:
                values = [val.replace('\ufeff', '') for val in values.split(";")]
                if len(values) > 1:
                    self.bn[key] = values

        self.bn['rv_key_values'] = rv_key_values
        print("TOKENISED key-values=" + str(self.bn))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: BayesNetReader.py [your_config_file.txt]")
    else:
        file_name = sys.argv[1]
        BayesNetReader(file_name)