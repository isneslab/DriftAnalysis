"""
visual.py
~~~~~~~~~~~

This module loads handles all visual plotting.

"""
import sqlite3
import matplotlib as plt
import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import rcParams, rc
from colors import ColorMap

class ResultsLoader():
    def __init__(self):
        pass
    
    
    def query_database_for_ID(self, experiment, train_families, test_families, dataset):
        # Loads in results which I can then pass to a viz class for actual plotting

        # Establish connections
        conn = sqlite3.connect('pkl_files/results.db')
        c = conn.cursor()       
        
        condition = ""
        for family in train_families:
            condition += f"Train_family like '%{family}%' and "
        for family in test_families:
            condition += f"Test_family like '%{family}%' and "
        condition += f"Experiment = '{experiment}' and "
        condition += f"Dataset = '{dataset}'"
        
        query = "SELECT rowid, Date, Results from result WHERE {}".format(condition)
        c.execute(query)
        
        files = c.fetchall()
        conn.commit()
        conn.close()
        
        print("Found {} files matching query".format(len(files)))
        for n in files:
            print(n)

        
    def load_file_from_id(self, ID):
        conn = sqlite3.connect("pkl_files/results.db")
        c = conn.cursor()
        query = "SELECT Results from result WHERE rowid = {}".format(ID)
        c.execute(query)
        filename = c.fetchall()[0][0]
        # Add some information of what was loaded
        with open(f"pkl_files/{filename}", 'rb') as f:
            data = pickle.load(f)
        
        return data


class Viz():
    def __init__(self, results1, results2=None, label1='C1', label2='C2'):
        """Base visual class that set global matplotlib parameters and inputs
        used for plotting

        Args:
            results1 (_type_): _description_
            results2 (_type_, optional): _description_. Defaults to None.
            label1 (str, optional): _description_. Defaults to 'C1'.
            label2 (str, optional): _description_. Defaults to 'C2'.
        """        
        self.results1 = results1
        self.results2 = results2
        self.label1 = label1
        self.label2 = label2
        
        rc('font', **{'size': 20, 'family': 'serif', 'serif': ['Computer Modern Roman']})
        rc('text', usetex=True)
        rcParams['figure.figsize'] = [18.5, 9.5]
    
    def performance(self, ax):
        """Function that plots the performance curve of results1 and/or results2.
        If only 1 input is available, f1, precision and recall are all plotted. Else
        only f1 is plotted.

        Args:
            ax (matplotlib.pyplot.axis): Matplotlib axis to plot on
        """               
        # Find results1 performance metrics
        y1_f1 = self.results1['f1'][1:]
        y1_prec = self.results1['precision'][1:]
        y1_recall = self.results1['recall'][1:]
        
        # Set x to plot
        x = np.arange(len(y1_f1))
        alpha_1 = 1
        
        # Plot f1 score for results 2 and lower alpha of results 1 
        if self.results2 != None:   
            y2_f1 = self.results2['f1'][1:]
            ax.plot(x, y2_f1, color = ColorMap['F1'].value, markersize=7, marker="o", 
                       label=self.label2, linewidth=2)
            alpha_1 = 0.3
        else:
            ax.plot(x, y1_recall, color= ColorMap['RECALL'].value, markersize=7, 
                    marker="^", label=f'Recall {self.label1}', linewidth=2)
            
            ax.plot(x, y1_prec, color= ColorMap['PREC'].value, markersize=7, 
                    marker="s", label=f'Precision {self.label2}', linewidth=2)

        # Plot f1 score for results 1
        ax.plot(x, y1_f1, color = ColorMap['F1'].value, markersize = 7, marker="o",
                label=self.label1, linewidth=2, alpha=alpha_1)
        ax.fill_between(x, 0, y1_f1, facecolor='none', hatch='/',
                        edgecolor='#BCDEFE', rasterized=True)

        months_of_x = len(self.results1['total_family'])+1
        labels_x = [str(x-6) if ((x > 6 and ((x-1) % 5 == 0 or x == 7)) or x == 1)
                    else '' for x in range(1,months_of_x)]

        ax.set_xticks(np.arange(-6,months_of_x-7))
        ax.set_xticklabels(labels_x)
        ax.set_ylabel("Performance")
        ax.legend(loc='best', fontsize=15)
        ax.set_xlim(-6.5,months_of_x-7)
    
    def gradient_poi_selection(self,ax):
        # Gradient POI selection, clean this up later
        difference = np.subtract(self.results1['f1'],self.results2['f2'])
        diff = np.diff(difference)
        diff = np.insert(diff,0,0)
        result = np.gradient(difference)
        result = [n3 if n1 > n2 else 0 for n1,n2,n3 in zip(self.results1['f1'],self.results2['f2'], result)]
        result = np.array(result)
        
        # plotting
        markers_on = result.argsort(axis=0)[::-1][:3]
        x = np.arange(len(result))
        ax.plot(x, result,linewidth=2, alpha=0.6,label='Gradients',color='r')
        for n in markers_on:
            ax.axvline(n+7,linestyle='-.',color='r')
        ax.set_ylim(0,1)
      
    
    def distribution(self, ax, goodware=False):
        """Function that shows the distribution of families in the samples used
        for both training and testing. Training months are shifted to begin at -5 to 0 and
        testing months begin at 1. 
        # TODO: Automate training shift
        # TODO: Currently only shows first half as total_family records on first half

        Args:
            ax (matplotlib.pyplot.axis): Matplotlib axis to plot on
            goodware (bool, optional): Whether goodware should be shown. Defaults to False.
        """        
        families = self.get_families_used(goodware)
        groups = np.arange(-6, len(self.results1['total_family']) - 6)
        previous_count = np.array([0]* len(self.results1['total_family']))
        
        for family in families:
            count_by_family = []
            for group in groups:
                if family in self.results1['total_family'][group + 6]:
                    count_by_family.append(self.results1['total_family'][group + 6][family])
                else:
                    count_by_family.append(0)

            ax.bar(groups, count_by_family, label=family, bottom=previous_count,
                   color=ColorMap[family].value)
            previous_count += np.array(count_by_family)
        
        months_of_x = len(self.results1['total_family'])+1
        labels_x = [str(x-6) if ((x > 6 and ((x-1) % 5 == 0 or x == 7)) or x == 1)
                    else '' for x in range(1,months_of_x)]
        
        ax.set_xticks(np.arange(-6,months_of_x-7))
        ax.set_xticklabels(labels_x)
        ax.set_ylabel('\# of samples')
        ax.legend(loc='best', fontsize=15)
        ax.set_xlim(-6.5,months_of_x-7)
    
    def get_families_used(self, goodware=True):
        """Helper function that iterates through all months and returns
        the families used in both training and testing

        Args:
            goodware (bool, optional): Whether goodware should be shown. Defaults to True.

        Returns:
            list: List of families used
        """        
        output = []
        for n in self.results1['total_family']:
            for family in list(n.keys()):
                if family not in output:
                    if goodware == False and family != 'GOODWARE':
                        output.append(family)
                    elif goodware:
                        output.append(family)
                    else:
                        pass
        return output
    
    
    def family_diff(self, ax, month_selection=None):
        """Function that plots the difference between true positives of results1 and results2.

        Args:
            ax (matplotlib.pyplot.axis): Matplotlib axis to plot on
            month_selection (list, optional): List of months to plot. If None, then all months are plotted.
            
        """        
        if self.results1 == None or self.results2 == None:
            print("Results 1 or Results 2 is None")
            return None 
        
        
        # Get all families 
        familes = self.get_families_used(goodware=True)
        previous_count = [0] * len(self.results1['Correct'])
        
        # Loop through families
        for family in familes:
            results1_true_positives = []
            results2_true_positives = []
            
            # Get a count of individial families for every group
            for group in range(len(self.results1['Correct'])):
                if family in self.results1['Correct'][group]:
                    results1_true_positives.append(\
                        self.results1['Correct'][family])
                    
                if family in self.results2['Correct'][group]:
                    results2_true_positives.append(\
                        self.results2['Correct'][family])
            
            # Find the difference between results2 and results 1 per family
            true_positive_diff = np.subtract(results2_true_positives,\
                                             results1_true_positives)
            
            # Normalise results to total number of samples per group
            true_positive_diff_norm = np.divide(true_positive_diff,\
                                                self.results1['total_family'][6:])
            
            # Check if month selection is given
            if month_selection != None:
                X = np.arange(len(month_selection))
                true_positive_diff_norm = true_positive_diff_norm[month_selection]
                if len(previous_count) != len(month_selection):
                    previous_count[month_selection]
            else:
                X = np.arange(len(self.results1['Correct']))
            
            # Plot bar plot
            ax.bar(X, true_positive_diff_norm, color=ColorMap[family].value,\
                  label = family.capitalize(), bottom=previous_count)
            
            previous_count = true_positive_diff_norm

        # Plot parameters
        ax.set_xticks(len(X))
        if month_selection == None:
            labels_x = [str(x-6) if ((x > 6 and ((x-1) % 5 == 0 or x == 7)) or x == 1)
                    else '' for x in range(1,len(X))]
        else:
            labels_x = np.add([1] * len(X), month_selection)
        ax.set_xticklabels(labels_x)
        ax.set_ylabel(f"Difference of true positives\nof \
                      {self.label1} and {self.label2} normalised")
        ax.set_ylim(0,1)
        ax.legend(loc='best', fontsize=15)

    
    def tsne(self):
        # tsne plots
        # maybe put this in seperate class
        pass
    
    def family_matrix(self):
        # family train on 1 test on rest plots
        # maybe put this in seperate class
        pass
    
    
    def plot_performance_distribution(self, poi=True):
        """Function that creates performance plot on the top and distribution
        plot on the bottom
        
        Args:
            poi (bool, optional): Whether POI should be plotted on distribution graph
        """        
        fig, ax = plt.subplots(2,1)
        
        self.performance(ax[0])
        self.distribution(ax[1])      
                
        if self.results2 != None and poi == True:
            ax2 = ax[1].twinx()
            self.gradient_poi_selection(ax2)

            ax2.set_ylabel("Gradient of difference")
        
        plt.xlabel('Month')
        plt.grid(visible=True, which='major', axis="y")
        plt.show()
    
    def plot_single(self, plot, month_selection=None):
        """Function that creates one plot, takes in a plot type of 'distribution',
        'performance','difference'

        Args:
            plot (str): Type of plot to plot
            month_selection (list, optional): List of months to plot, if None then all months plotted.

        """        
        fig, ax = plt.subplots(1)
        
        
        if plot.lower() == 'distribution':
            self.distribution(ax)
        elif plot.lower() == 'performance':
            self.performance(ax)
        elif plot.lower() == 'difference':
            self.family_diff(ax, month_selection)
        else:
            print("Plot type not found, try: \
                  distribution|performance|difference")
            return None
        
        plt.xlabel('Month')
        plt.grid()
        plt.show()
        
        
class VizExpl():
    def __init__(self, results1, results2=None, label1='C1',label2='C2'):
        self.results1 = results1
        self.results2 = results2
        self.label1 = label1
        self.label2 = label2

        
    def get_explanations_mean(self, expl, filter_list):
        output = []
        # Take the mean of every sample
        for group in len(expl):
            if isinstance(expl[group], np.ndarray):
                if filter_list != []:
                    explanations = expl[group][filter_list]
                else:
                    explanations = expl[group]
                output.append(np.mean(explanations, axis=0))
            else:
                output.append(0)
    
        return output
    
    def get_top_features(self, result, family_select, k=5, group_selection=None):
        # TODO: Add family selection
        filter_list = []
        if family_select != None:
            for group in len(result['family_class']):
                filter_group = []
                for sample in group:
                    if sample in family_select:
                        filter_group.append(sample)
            
                filter_list.append(filter_list)
        
        expl_mean = self.get_explanations_mean(result['explanation'], filter_list)
        output = []
        
        if group_selection == None:
            group_selection = np.arange(len(expl_mean))        
        
        for group in group_selection:
            topk_filter = expl_mean[group].argsort(axis=0)[::-1][:k]
            output.append(topk_filter)
            
            # Print results
            print('-'*20)
            for idx, n in enumerate(topk_filter):
                print('{}   {}  {}'.format(idx, result['feature_names'][n], n))
                
        return output
    
    def feature_difference(self, k=30, group_selection=None):
        if self.results1 == None or self.results2 == None:
            print("Results 1 or Results 2 is None")
            return None 
        
        topk_features1 = self.get_top_features(self.results1, k=k, group_selection=group_selection)
        topk_features2 = self.get_top_features(self.results2, k=k, group_selection=group_selection)
        
        for group in len(topk_features1):
            results1_topk_set = set(topk_features1[group])
            results2_topk_set = set(topk_features2[group])
            
            new_features  = results2_topk_set - results1_topk_set
            
            # Print results
            print('-'*20)
            for idx, n in enumerate(new_features):
                print('Group {}'.format(group))
                print('-'*5)
                print('{}   {}  {}'.format(idx, self.results1['feature_names'][n], n))     
    
    
    
    
    
if __name__=='__main__':
    print("Visual module")
    training_familes = ['Dowgin','Dnotua','Kuguo','Airpush','Revmob']
    testing_families = ['Dowgin','Dnotua','Kuguo','Airpush','Revmob']
    ResultsLoader().query_database_for_ID('half',training_familes,testing_families,'Transcend')
    base = ResultsLoader().load_file_from_id(5)
    # visual = Viz(base)
    # visual.plot_performance_distribution()
    # visual.plot_single('distribution')
    
    visual = VizExpl(base)
    visual.get_explanations_mean(base)