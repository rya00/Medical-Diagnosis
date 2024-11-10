class CSV_DataReader:
    def __init__(self, file_name):
        if file_name is None:
            raise ValueError("Error: No file name provided.")
        
        # Initialize class attributes
        self.rand_vars = []
        self.rv_key_values = {}
        self.rv_all_values = []
        self.predictor_variable = None
        self.num_data_instances = 0

        # Read the data from the provided file
        self.read_data(file_name)

    def read_data(self, data_file):
        print("\nREADING data file %s..." % (data_file))
        print("---------------------------------------")

        try:
            with open(data_file, encoding='UTF-8') as csv_file:
                first_line = True  # Track if we're reading the header
                for line in csv_file:
                    line = line.strip().replace('\ufeff', '')
                    if len(line) == 0:
                        continue  # Skip empty lines

                    values = line.split(',')
                    if first_line:
                        # Read the header line
                        self.rand_vars = [var.replace('\ufeff', '').strip() for var in values]
                        for variable in self.rand_vars:
                            self.rv_key_values[variable] = []
                        first_line = False
                    else:
                        # Read the data lines
                        self.rv_all_values.append(values)
                        self.update_variable_key_values(values)
                        self.num_data_instances += 1

            # Set the predictor variable
            self.predictor_variable = self.rand_vars[-1]

            # Debugging outputs
            print("RANDOM VARIABLES=%s" % (self.rand_vars))
            print("VARIABLE KEY VALUES=%s" % (self.rv_key_values))
            print("VARIABLE VALUES=%s" % (self.rv_all_values))
            print("PREDICTOR VARIABLE=%s" % (self.predictor_variable))
            print("|data instances|=%d" % (self.num_data_instances))

        except FileNotFoundError:
            print(f"Error: The file '{data_file}' was not found.")
        except Exception as e:
            print(f"Error reading data file: {e}")

    def update_variable_key_values(self, values):
        for i, variable in enumerate(self.rand_vars):
            if i < len(values):  # Ensure the index is within bounds
                value_in_focus = values[i]
                if value_in_focus not in self.rv_key_values[variable]:
                    self.rv_key_values[variable].append(value_in_focus)

    def get_true_values(self):
        """ Extract the true values for the predictor variable. """
        if self.predictor_variable is None:
            print("No predictor variable set. Cannot retrieve true values.")
            return []

        try:
            predictor_index = self.rand_vars.index(self.predictor_variable)
            true_values = [row[predictor_index] for row in self.rv_all_values if len(row) > predictor_index]
            return true_values
        except Exception as e:
            print(f"Error retrieving true values: {e}")
            return []
