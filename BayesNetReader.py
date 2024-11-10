import sys


class BayesNetReader:
    bn = {}

    def __init__(self, file_name):
        self.read_data(file_name)
        self.tokenise_data()

    def read_data(self, data_file):
        print("\nREADING data file %s..." % (data_file))

        with open(data_file) as cfg_file:
            key = None
            value = None
            for line in cfg_file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split(":")
                if len(tokens) == 2:
                    if value is not None:
                        self.bn[key] = value
                        value = None

                    key = tokens[0]
                    value = tokens[1]
                else:
                    value += tokens[0]

        self.bn[key] = value
        self.bn["random_variables_raw"] = self.bn["random_variables"]
        print("RAW key-values="+str(self.bn))

    def tokenise_data(self):
        print("TOKENISING data...")
        rv_key_values = {}

        for key, values in self.bn.items():

            if key == "random_variables":
                var_set = []
                for value in values.split(";"):
                    if value.find("(") and value.find(")"):
                        value = value.replace('(', ' ')
                        value = value.replace(')', ' ')
                        parts = value.split(' ')
                        var_set.append(parts[1].strip())
                    else:
                        var_set.append(value)
                self.bn[key] = var_set

            elif key.startswith("CPT"):
                # store Conditional Probability Tables (CPTs) as dictionaries
                cpt = {}
                sum = 0
                for value in values.split(";"):
                    pair = value.split("=")
                    cpt[pair[0]] = float(pair[1])
                    sum += float(pair[1])
                print("key=%s cpt=%s sum=%s" % (key, cpt, sum))
                self.bn[key] = cpt

                # store unique values for each random variable
                if key.find("|") > 0:
                    rand_var = key[4:].split("|")[0]
                else:
                    rand_var = key[4:].split(")")[0]
                unique_values = list(cpt.keys())
                rv_key_values[rand_var] = unique_values

            else:
                values = values.split(";")
                if len(values) > 1:
                    self.bn[key] = values

        self.bn['rv_key_values'] = rv_key_values
        print("TOKENISED key-values="+str(self.bn))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: BayesNetReader.py [your_config_file.txt]")
    else:
        file_name = sys.argv[1]
        BayesNetReader(file_name)
