import sys
import time
import pickle
import numpy as np
from BayesNetReader import BayesNetReader
from DataReader import CSV_DataReader
from sklearn.linear_model import Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV


class PDF_Generator(BayesNetReader):
    means = {} # mean vectors of random variables
    stdevs = {} # standard deviations of random variables
    regressors = {} # regression models for random variables
    REGRESSOR2EMPLOY = 'LASSO' # choices are 'RIDGE', 'LASSO' or 'KERNELRIDGE'

    def __init__(self, configfile_name, datafile_name):
        # Bayes net reading (configfile_name in text format)
        # and data loading (datafile_name in csv format)
        self.bn = BayesNetReader(configfile_name)
        self.csv = CSV_DataReader(datafile_name)

        # model training and saving, which updates the text file
        # configfile_name and generates an equivalent *.pkl file 
        self.running_time = time.time()
        self.estimate_regression_models()
        self.update_configuration_file(configfile_name)
        self.running_time = time.time() - self.running_time
        print("Training Time="+str(self.running_time)+" secs.")

    # computes the following for each random variable in the network:
	# (a) mean and standard deviation
    # (b) regression models via GradientBoostingRegressor,
    #     which you can change to the method of your choice.
    def estimate_regression_models(self):
        print("\nESTIMATING %s regression models..." % (self.REGRESSOR2EMPLOY))
        print("---------------------------------------------------")

        for pd in self.bn.bn["structure"]:
            print(str(pd))
            p = pd.replace('(', ' ')
            p = p.replace(')', ' ')
            tokens = p.split("|")

            # estimate mean and standard deviation per random variable
            variable = tokens[0].split(' ')[1]
            feature_vector = self.get_feature_vector(variable)
            self.means[variable] = np.mean(feature_vector)
            self.stdevs[variable] = np.std(feature_vector)
            #print("mean=%s stdev=%s" % (self.means[variable], self.stdevs[variable]))

            # train regression models via a GradientBoostingRegressor
            if len(tokens) == 2:
                variable = tokens[0].split(' ')[1]
                parents = tokens[1].strip().split(',')
                inputs, outputs = self.get_feature_vectors(parents, variable)
                regression_model = self.get_optimised_regression_model(inputs, outputs)
                regression_model.fit(inputs, outputs)
                self.regressors[variable] = regression_model
                print("Created regression model for variable %s\n" % (variable))
            else:
                print("Estimated means and stdevs for variable %s\n" % (variable))

    def get_optimised_regression_model(self, inputs, outputs):
        # grids for alpha and gamma hyperparameters
        alpha = [0.01, 0.1, 0.5, 1, 2, 3, 5, 10]  # Regularization parameter
        gamma = [0.01, 0.1, 0.5, 1, 2, 3, 5, 10]  # Parameter for RBF kernel
        max_iter = [1000, 5000, 10000] # max. number of iterations by solver
        tol = [1e-4, 1e-5, 1e-6]  # amount of tolerance for convergence

        # focus on RBF kernel only -- others are possible but not attempted here
        if self.REGRESSOR2EMPLOY == 'RIDGE':
            regression_model = Ridge()
            parameter_grid = {'alpha': alpha, 'max_iter': max_iter, 'tol': tol}
        elif self.REGRESSOR2EMPLOY == 'LASSO':
            regression_model = Lasso()
            parameter_grid = {'alpha': alpha, 'max_iter': max_iter, 'tol': tol}
        elif self.REGRESSOR2EMPLOY == 'KERNELRIDGE':
            regression_model = KernelRidge(kernel='rbf') 
            parameter_grid = {'alpha': alpha, 'gamma': gamma}
        else:
            print("UNKNOWN REGRESSOR2EMPLOY="+str(self.REGRESSOR2EMPLOY))
            exit(0)

        # make use of GridSearchCV to find the best hyperparameter values using cross-validation (cv)
        grid_search = GridSearchCV(regression_model, parameter_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(inputs, outputs)

        # retrieve the best alpha, gamma and model
        best_params = grid_search.best_params_
        print("Best hyperparameters: %s" % (best_params))

        if self.REGRESSOR2EMPLOY == 'RIDGE':
            best_model = Ridge(alpha=best_params['alpha'], max_iter=best_params['max_iter'])
        elif self.REGRESSOR2EMPLOY == 'LASSO':
            best_model = Lasso(alpha=best_params['alpha'], max_iter=best_params['max_iter'])
        elif self.REGRESSOR2EMPLOY == 'KERNELRIDGE':
            best_model = KernelRidge(kernel='rbf', alpha=best_params['alpha'], gamma=best_params['gamma'])

        return best_model.fit(inputs, outputs)

    # returns the data column of the random variable given as argument
    def get_feature_vector(self, variable):
        variable_index = self.get_variable_index(variable)
        feature_vector = []
        counter = 0
        for datapoint in self.csv.rv_all_values:
            value = datapoint[variable_index]
            feature_vector.append(value)

        return np.asarray(feature_vector, dtype="float32")

    # returns the index (0 to N-1) of the random variable given as argument
    def get_variable_index(self, variable):
        for i in range(0, len(self.csv.rand_vars)):
            if variable == self.csv.rand_vars[i]:
                return i
        print("WARNING: couldn't find index of variables=%s" % (variable))
        return None

    # return the data columns of the parent random variables given as argument
    def get_feature_vectors(self, parents, variable):
        input_features = []
        for parent in parents:
            feature_vector = self.get_feature_vector(parent)
            if len(input_features) == 0:
                for f in range(0, len(feature_vector)):
                    input_features.append([feature_vector[f]])
            else:
                for f in range(0, len(feature_vector)):
                    tmp_vector = input_features[f]
                    tmp_vector.append(feature_vector[f])
                    input_features[f] = tmp_vector

        output_features = self.get_feature_vector(variable)

        input_features = np.asarray(input_features, dtype="float32")
        output_features = np.asarray(output_features, dtype="float32")
        input_features, output_features = self.restrict_data(input_features, output_features)
        return input_features, output_features

    # restrict the data, but only if it is too much in particular for KernelRidge regression
    def restrict_data(self, inputs, outputs):
        MAX_TRAIN_DATA = 2000
        isLargeData4KernelTrick = True if len(inputs) > MAX_TRAIN_DATA else False
        if self.REGRESSOR2EMPLOY == 'KERNELRIDGE' and isLargeData4KernelTrick:
            random_indices = np.random.choice(inputs.shape[0], MAX_TRAIN_DATA, replace=False)
            inputs = np.asarray(inputs[random_indices], dtype="float32")
            outputs = np.asarray(outputs[random_indices], dtype="float32")
            print("DATA restricted to %s instances" % (len(inputs)))
        return inputs, outputs

    # re-writes the provided configuration file with information about
    # regression models -- one per random variable in the network.
    # The means, standard deviations and regression models are all
    # stored in a PICKLE file due for data saving/loading convience.
    # Such files with extension .pkl are stored in the config folder.
    def update_configuration_file(self, configfile_name):
        print("WRITING config file with regression models...")
        print("See rewritten file "+str(configfile_name))
        print("---------------------------------------------------")
        name = self.bn.bn["name"]

        rand_vars = self.bn.bn["random_variables_raw"]
        rand_vars = str(rand_vars).replace('[', '').replace(']', '')
        rand_vars = str(rand_vars).replace('\'', '').replace(', ', ';')

        structure = self.bn.bn["structure"]
        structure = str(structure).replace('[', '').replace(']', '')
        structure = str(structure).replace('\'', '').replace(', ', ';')
		
        regression_models = {}
        regression_models['means'] = self.means
        regression_models['stdevs'] = self.stdevs
        regression_models['regressors'] = self.regressors
        regression_models_filename = configfile_name[:len(configfile_name)-4]
        regression_models_filename = regression_models_filename+'.pkl'

        with open(configfile_name, 'w') as cfg_file:
            cfg_file.write("name:"+str(name))
            cfg_file.write('\n')
            cfg_file.write('\n')
            cfg_file.write("random_variables:"+str(rand_vars))
            cfg_file.write('\n')
            cfg_file.write('\n')
            cfg_file.write("structure:"+str(structure))
            cfg_file.write('\n')
            cfg_file.write('\n')
            cfg_file.write("regression_models:"+str(regression_models_filename))

        with open(regression_models_filename, 'wb') as models_file:
            pickle.dump(regression_models, models_file)


if len(sys.argv) != 3:
    print("USAGE: PDF_Generator.py [your_config_file.txt] [training_file.csv]")
    print("EXAMPLE> PDF_Generator.py config_banknote_authentication.txt data_banknote_authentication-train.csv")
    exit(0)

else:
    configfile_name = sys.argv[1]
    datafile_name = sys.argv[2]
    PDF_Generator(configfile_name, datafile_name)
