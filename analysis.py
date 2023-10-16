"""
analysis.py
~~~~~~~~~~~

A module for the main experiment which calculates the
performance for each month. Saving the results in a
pickle format.


"""
import numpy as np
from sklearn import svm
from sklearn.utils import shuffle
import pickle
from collections import Counter
import sqlite3
from datetime import datetime
import random

from secml.ml.classifiers import CClassifierSVM
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.manifold import TSNE

import explanations
from util import get_random_string


class Analysis():    
    def __init__(self, X, y,t,f, feature_names, md5, train_index, test_index):       
        """Initialise analysis class with input data and indexes 

        Args:
            X (np.ndarray): Multi-dimensinal array of predictors
            y (np.ndarray): Array of output labels
            t (np.ndarray): Array of timestamp tags
            f (np.ndarray): Array of family labels
            feature_names (np.ndarray): Array of feature names
            train_index (np.ndarray): Array of indexes for training set
            test_index (np.ndarray): Array of array of indexes for test set
        """        
        self.X = X
        self.y = y
        self.t = t
        self.f = f
        self.feature_names = feature_names
        self.md5 = md5
        self.initial_train = train_index
        self.initial_test = test_index
        self.results = {"accuracy":[], "f1":[], "recall":[], "precision":[], "f1_n":[],  "recall_n":[], "precision_n":[],
                        "train_amount":[], "test_amount":[], "total_family":[], "explanations":[], "correct_family":[], 
                        "family_class":[], "feature_names":feature_names}
        
    
    def train_svc_model(self,X_train, y_train):
        model = svm.SVC(C = 1.0, kernel='linear')
        model.fit(X_train, y_train)
        return model

    def train_secml_model(self, X_train, y_train):
        model = CClassifierSVM()
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self,model, X_test, y_test):
        y_pred = model.predict(X_test)
        if type(y_pred).__module__ != np.__name__:
            y_pred = y_pred.get_data()
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        precision = precision_score(y_test, y_pred, pos_label=1)
        f1_n = f1_score(y_test, y_pred, pos_label=0)
        recall_n = recall_score(y_test, y_pred, pos_label=0)
        precision_n = precision_score(y_test, y_pred, pos_label=0)

        return accuracy, f1, recall, precision, f1_n, recall_n, precision_n,y_pred
    
    def update_performance_results(self, acc, f1_score, recall, precision, 
                                   f1_score_n, recall_n, precision_n, train_amt, test_amt, expl=0):
        """Updates performance results for export"""    
            
        self.results['accuracy'].append(acc)
        self.results['f1'].append(f1_score)
        self.results['recall'].append(recall)
        self.results['precision'].append(precision)
        self.results['f1_n'].append(f1_score_n)
        self.results['recall_n'].append(recall_n)
        self.results['precision_n'].append(precision_n)
        self.results['train_amount'].append(train_amt)
        self.results['test_amount'].append(test_amt)
        self.results['explanations'].append(expl)
        
    def update_family_results(self, total_samples, correct_samples, family_class):
        """Update family specific results"""
        
        # Total samples in month, including first and second half
        self.results['total_family'].append(total_samples)
        # Dictionary of number of correct samples per family
        self.results['correct_family'].append(correct_samples)
        # List of predicted and family lables of tested samples, only first half
        self.results['family_class'].append(family_class)
        
    def check_family_labels(self, preds, indexes, all_indexes=None):
        """Function that checks total family count and true
        positive count for each family for a given list
        of samples. Then updates then to results.

        Args:
            preds (np.ndarray): Array of predictions
            indexes (list): List of indexes for samples
            all_indexes (list, optional): List of all indexes in 1 month, this
                        is for logging number of samples in a month before split. 
        """        
        
        if all_indexes == None:
            total_family_count = Counter(self.f[indexes])
        else:
            total_family_count = Counter(self.f[all_indexes])
                
        true_positives = {'GOODWARE':0}
        family_class = []
        for idx, pred in enumerate(preds):
            family_class.append([pred, self.f[indexes[idx]]])
            # Check if predicted malware and ground truth is malware
            if pred == 1 and self.y[indexes[idx]] == 1:
                sample_family = self.f[indexes[idx]]
                # Increment true positive count for each family
                if sample_family not in true_positives:
                    true_positives[sample_family] = 1
                else:
                    true_positives[sample_family] += 1
        
            # Check if predicted goodware and ground truth is goodware
            if pred == 0 and self.y[indexes[idx]] == 0:
                true_positives['GOODWARE'] += 1
                
        self.update_family_results(total_family_count, true_positives, family_class)
                 
        
    def save_results(self, experiment, training_family, testing_family, dataset):
        """Function that save results and logs them in a MYSQL database for easy lookup

        Args:
            experiment (str): Experiment type
            training_family (list): List of families used during training
            testing_family (list): List of families used during testing
            dataset (str): Dataset used for experiment
        """        
        # Call create database to create database if does not exist
        self.create_database()
        
        # Establish connections
        conn = sqlite3.connect('pkl_files/results.db')
        c = conn.cursor()
        
        # Log relavent family
        training_family_string = ' '.join(training_family)
        testing_family_string = ' '.join(testing_family)
        
        # Organise results info as dictionary
        data = {'Date' : datetime.now(),
                'Experiment' : experiment,
                'Dataset' : dataset,
                'Train_family': training_family_string,
                'Test_family': testing_family_string
                }
  
        # Export explanations to file and save file name
        random_file_name = get_random_string(10)
        random_file_name += '.pkl'
        with open(f"pkl_files/{random_file_name}",'wb') as f:
            pickle.dump(self.results,f)
        data['Results'] = random_file_name
            
        # Convert data dict to query command
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data.values()])
        query = f"INSERT INTO result ({columns}) VALUES ({placeholders})"
        values = tuple(data.values())
            
        # Execute and commit
        c.execute(query, values)
        lastrowid = c.lastrowid
        print(f"Saved result with ID {lastrowid}")
        conn.commit()
        conn.close()

        return lastrowid

    def create_database(self):
        """Function that creates a sqlite database if one
        does not exist already. This database is used to save
        all results.
        """        
        conn = sqlite3.connect('pkl_files/results.db')
        c = conn.cursor()
        
        columns_string = "Date text, Experiment text, Dataset text,\
                          Results text, Train_family text, Test_family text"
            
                        

        c.execute("CREATE TABLE if not exists result ({})".format(columns_string))
        
    
        conn.commit()
        conn.close()  
        
    def get_family_count_in_training(self, indexes):
        # This is simply for visualisation and WILL cause errors if
        # different time windows are set. Currently it splits the training
        # set in to 6 months and counts up the families returning a dict. Any
        # months greater than 6 is merged to month 6
        
        output = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
        for index in indexes:
            if self.t[index].month > 6:
                output[6].append(self.f[index])
            else:
                output[self.t[index].month].append(self.f[index])
            
        final_output = []
        for month in range(1,7):
            self.results['total_family'].append(Counter(output[month]))     
            final_output.append(output[month])
        
        return final_output
    
    def training(self, family=None, additional_months=False):
        """Function that trains a model using initial training set
        TODO: Add arguments for adding extra samples depending on family and month
        
        Args:
            family (list, optional): List of families selected for training, if None then
            all familes are used.
            additional_months (bool, optional): Whether to add additional training months

        Returns:
            object: Trained model
            int: Number of samples used for training
        """        
        
        # Select samples of chosen families   
        selected_indexes = []
        for train_group in range(len(self.initial_train)):
            if family != None:
                family = list(map(str.upper,family))
                selected_indexes += self.family_selection_from_index(self.initial_train[train_group],family)
            else:
                selected_indexes += self.initial_train[train_group]
        
        # Add extra training samples from test set
        if additional_months:
            selected_indexes += self.family_selection_from_index(self.initial_test[31],['DNOTUA'])
            selected_indexes += self.family_selection_from_index(self.initial_test[32],['DNOTUA'])
            selected_indexes += self.family_selection_from_index(self.initial_test[33],['DNOTUA'])
            selected_indexes += self.family_selection_from_index(self.initial_test[34],['DNOTUA'])
            selected_indexes += self.family_selection_from_index(self.initial_test[35],['DNOTUA'])
            selected_indexes += self.family_selection_from_index(self.initial_test[36],['DNOTUA'])
                
        model = self.train_secml_model(self.X[selected_indexes], self.y[selected_indexes])

        # Evaluate trained model
        *eval_out, = self.evaluate_model(model, self.X[selected_indexes], self.y[selected_indexes])
        print("Trained model with initial training set of length {}".format(len(selected_indexes)))
        print("Achieved accuracy of {0:3f} and f1 score of {1:3f}".format(eval_out[0], eval_out[1]))

        
        # Set up explainer and compute explanations
        explainer = explanations.Explain()
        explainer.set_input_data(model, self.X[selected_indexes], self.y[selected_indexes])
        explanations_results = explainer.IG()
        
        # Get total families for each month in training set and update it. Also family labels
        family_labels_array = self.get_family_count_in_training(selected_indexes)
        # Update results
        self.update_performance_results(*eval_out[:-1], len(selected_indexes),
                                        family_labels_array,[explanations_results, self.md5[selected_indexes]])
        
        
        
        return model, len(selected_indexes)
    
    def testing(self, model, training_amount, family=None):
        """Experiment that tests on all samples of ever group

        Args:
            model (object): Pretrained model
            training_amount (int): Number of samples used when training initial model
            family (list, optional): List of families selected for training, if None then
            all familes are used.
        """                
        explainer = explanations.Explain()
        for group in range(len(self.initial_test)):
            # Select test samples
            if family != None:
                family = list(map(str.upper,family))
                selected_indexes = self.family_selection_from_index(self.initial_test[group],family)
            else:
                selected_indexes = self.initial_test[group]
            
            X_test = self.X[selected_indexes]
            y_test = self.y[selected_indexes]
            
            # Evaluate trained model
            *eval_out, = self.evaluate_model(model, X_test, y_test)
            print("Tested on group {} with {} training samples and {} testing samples"
                  .format(group, training_amount, len(y_test)))

            # Compute explanations
            explainer.set_input_data(model, X_test, y_test)
            explanations_results = explainer.IG()
        
            # Update results
            self.update_performance_results(*eval_out[:-1], training_amount,
                                        self.f[selected_indexes],[explanations_results, self.md5[selected_indexes]])
            self.check_family_labels(eval_out[-1], selected_indexes)
        
    
    def family_selection_from_index(self, indexes, family, goodware=True):
        """Helper function to select all families from 
        a given list of index

        Args:
            indexes (list): List of indexes selected
            family (list): List of families

        Returns:
            List: Indexes for a chosen family
        """        
        
        output = []
        for index in indexes:
            if self.f[index] in family:
                output.append(index)
                
        len_malware_samples = len(output)
        
        if goodware:
            output += self.goodware_selection_from_index(indexes, len_malware_samples)
            
        return output   
    
    def family_selection_from_index_recreate(self, indexes,families, goodware=True):
        output = []

        for family in families:
            for index in indexes:
                if self.f[index] == family:
                    output.append(index)
                    
        len_malware_samples = len(output)

        if goodware:
            output += self.goodware_selection_from_index(indexes, len_malware_samples)

        return output   



    def goodware_selection_from_index(self, indexes, amount):
        """Helper function to select goodware from a given list of index

        Args:
            indexes (list): List of indexes selected
            amount (int): Number of goodware to select

        Returns:
            List: Indexes of goodware samples
        """        
        output = []
        for index in indexes:
            if self.f[index] == 'GOODWARE':
                output.append(index)

                if len(output) >= amount:  
                    break
                
        return output
        
    def stratified_split(self, indexes, gw_split1=True, gw_split2=True):
        """Helper function that does straified split. Has option to add/remove
        goodware in first/second half.

        Args:
            indexes (list): List of selected indexes

        Returns:
            List: Indexes for first half of split
            List: Indexes for second half of split
        """        
        
        first_half = []
        second_half = []
        
        families = list(Counter(self.f[indexes]).keys())
    
        # Stratified split for families
        for family in families:
            if family != 'GOODWARE':
                full_family = self.family_selection_from_index(indexes, family, goodware=False)
                half = len(full_family)//2
                first_half += full_family[:half]
                second_half += full_family[half:]
        
        # Stratified split for goodware
        total_malware_samples = len(first_half) + len(second_half)
        goodware_samples = self.goodware_selection_from_index(indexes,total_malware_samples)
        if gw_split1:
            first_half += goodware_samples[:total_malware_samples//2]
        
        if gw_split2:
            second_half += goodware_samples[total_malware_samples//2:]
            
        return first_half, second_half
    
    def random_split(self, indexes, gw_split1=True, gw_split2=True):
        """Helper function that does random split. Has option to add/remove
        goodware in first/second half.

        Args:
            indexes (list): List of selected indexes

        Returns:
            List: Indexes for first half of split
            List: Indexes for second half of split
        """        
        
        total_number_of_samples = len(indexes)
        # Random split malware
        all_samples = shuffle(indexes, random_state=3)
        first_half = all_samples[:total_number_of_samples//2]
        second_half = all_samples[total_number_of_samples//2:]
        
        # Filter out goodware depending on arguments
        if gw_split1:
            first_half_filtered = first_half
        else:
            first_half_filtered = []
            for idx in first_half:
                if self.y[idx] == 1:
                    first_half_filtered.append(idx)
        
        if gw_split2:
            second_half_filtered = second_half
        else:
            second_half_filtered = []
            for idx in second_half:
                if self.y[idx] == 1:
                    second_half_filtered.append(idx)
        
            
        return first_half_filtered, second_half_filtered
    
    
    def testing_half_group(self, model, training_amount, family=None, strat_split=True):
        """Experiment that tests on half of each group. Stratified
        Sampling is used to ensure even split in families. However, there is option to turn
        off stratified sampling which recreates paper results using random split.

        Args:
            model (object): Pretrained model
            training_amount (int): Number of samples used when training initial model
            family (list, optional): List of families selected for training, if None then
            all familes are used.
        """        
        explainer = explanations.Explain()
        for group in range(len(self.initial_test)):
            # Select test samples
            if family != None:
                family = list(map(str.upper,family))
                selected_indexes = self.family_selection_from_index(self.initial_test[group],family)
            else:
                selected_indexes = self.initial_test[group]
            
            if strat_split:
                first_half, _ = self.stratified_split(selected_indexes)
            else:
                selected_indexes = self.family_selection_from_index_recreate(self.initial_test[group],family)
                first_half, _ = self.random_split(selected_indexes)
            
            X_test = self.X[first_half]
            y_test = self.y[first_half]         
            
            
            # Evaluate trained model
            *eval_out, = self.evaluate_model(model, X_test, y_test)
            print("Tested on group {} with {} training samples and {} testing samples"
                  .format(group, training_amount, len(y_test)))

            # Compute explanations
            explainer.set_input_data(model, X_test, y_test)
            explanations_results = explainer.IG()
        
            # Update results
            self.update_performance_results(*eval_out[:-1], training_amount,
                                        self.f[selected_indexes],[explanations_results, self.md5[first_half]])
            self.check_family_labels(eval_out[-1], first_half, selected_indexes)

        
    
    def testing_half_group_snooped(self, family=None, gw_split2=True, strat_split = True):
        """Experiment that trains on initial training + second half of
        testing for each group. Tests on remaining half the same group.
        StratifiedSampling is used to ensure even split in families.
        
        Args:
            family (list, optional): List of families selected for training, if None then
            all familes are used.
        """              
        family = list(map(str.upper,family))
        train_selected_indexes = [] 
        for train_group in range(len(self.initial_train)):
            train_selected_indexes += self.family_selection_from_index(self.initial_train[train_group],family)
          
        explainer = explanations.Explain()
        for group in range(len(self.initial_test)):
            # Select test samples
            if family != None:
                family = list(map(str.upper,family))
                selected_indexes = self.family_selection_from_index(self.initial_test[group],family)
            else:
                selected_indexes = self.initial_test[group]
            
            if strat_split:
                first_half, second_half = self.stratified_split(selected_indexes, gw_split2=gw_split2)
            else:
                selected_indexes = self.family_selection_from_index_recreate(self.initial_test[group],family)
                first_half, second_half = self.random_split(selected_indexes, gw_split2=gw_split2)
                
            X_test = self.X[first_half]
            y_test = self.y[first_half]

            # Snoop ahead and train on snooped group
            retrain_index = train_selected_indexes + second_half
            model = self.train_secml_model(self.X[retrain_index], self.y[retrain_index])
            print("Snooped on group {} with {} training samples".format(group, len(retrain_index)))

            
            # Evaluate trained model
            *eval_out, = self.evaluate_model(model, X_test, y_test)
            print("Tested on group {} with {} training samples and {} testing samples"
                  .format(group, len(retrain_index), len(y_test)))

            # Compute explanations
            explainer.set_input_data(model, X_test, y_test)
            explanations_results = explainer.IG()
        
            # Update results
            self.update_performance_results(*eval_out[:-1], len(retrain_index),
                                        self.f[selected_indexes],[explanations_results, self.md5[first_half]])
            self.check_family_labels(eval_out[-1], first_half, selected_indexes)
        
        
    def tsne(self, families):
        random.seed(10)
        families = list(map(str.upper, families))
        for family in families:
            tsne_final_result = []
            for group in range(len(self.initial_test)):
                selected_indexes = self.family_selection_from_index(self.initial_test[group], family,goodware=False)
                X = self.X[selected_indexes][:1000]

                if len(X) > 10:
                    tsne = TSNE(n_components=2, perplexity=10, learning_rate='auto', init='pca')
                    tsne_result = tsne.fit_transform(X)
                else:
                    tsne_result = []
                tsne_final_result.append(tsne_result)
                
            with open(f'pkl_files/tsne_{family}.pkl','wb') as f:
                pickle.dump(tsne_final_result,f)
        
        
    def run(self, training_family, testing_family, experiment, dataset):
        """Function that runs the corresponding experiment depending on the
        paramaters provided. Currently there are 3 experiments implemented. For
        detailed explanation refer to corresponding function.

        Args:
            training_family (np.array): Array of indexes of training samples
            testing_family (np.array): Array of array of indexes of test samples 
            experiment (str): Experiment to run
            dataset (str): Dataset being used, this is only for logging purposes
        """        
        # Main experiment that runs training and testing given certain params
        # then saves them in sqlite database
        
        available_experiments = ['base','half','snoop','nogwsnoop','half_random','snoop_random','nogwsnoop_random','base_additional']
        
        if experiment.lower() not in available_experiments:
            print("Error: Experiment not recognised, \
                  try: base|half|snoop|nogwsnoop|half_random|snoop_random|nogwsnoop_random|base_additional")
            exit()
            
        if experiment.lower() == 'base_additional':
            trained_model, trained_amount = self.training(training_family, additional_months=True)
            self.testing(trained_model, trained_amount, testing_family)
        else:
            trained_model, trained_amount = self.training(training_family)
            if experiment.lower() == 'base':
                self.testing(trained_model, trained_amount, testing_family)
            elif experiment.lower() == 'half':
                self.testing_half_group(trained_model, trained_amount,testing_family)
            elif experiment.lower() == 'snoop':
                self.testing_half_group_snooped(testing_family)
            elif experiment.lower() == 'nogwsnoop':
                self.testing_half_group_snooped(testing_family, gw_split2=False)
            elif experiment.lower() == 'half_random':
                self.testing_half_group(trained_model, trained_amount,testing_family, strat_split=False)
            elif experiment.lower() == 'snoop_random':
                self.testing_half_group_snooped(testing_family, strat_split=False)
            elif experiment.lower() == 'nogwsnoop_random':
                self.testing_half_group_snooped(testing_family, gw_split2=False, strat_split=False)
            else:
                pass
        
        if training_family == None:
            training_family = ['ALL']
        else:
            training_family = list(map(str.upper,training_family))
            
        if testing_family == None:
            testing_family = ['ALL']
        else:
            testing_family = list(map(str.upper,testing_family))
        
        lastrowid = self.save_results(experiment, training_family, testing_family, dataset)
        print("Done")
        return lastrowid