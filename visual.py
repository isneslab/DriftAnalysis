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
from matplotlib import colors, cm
from colors import ColorMap
import os


def save_output(fname="output.pdf"):
    fpath = os.path.join('results', fname)
    print(fpath)
    plt.savefig(fpath, bbox_inches='tight')

class DimensionReduction():
    def __init__(self):
        rc('font', **{'size': 20, 'family': 'serif', 'serif': ['Computer Modern Roman']})
        rc('text', usetex=True)
        rc('xtick', labelsize=15)
        rc('ytick', labelsize=15)
        rcParams['figure.figsize'] = [16, 7]
        
    def tsne_visual(self, families, fname='output.pdf'):
        fig, axs = plt.subplots(1,5)
        maxlim = (-1000,1000)
        
        colormap = plt.get_cmap('plasma')
        colors_list = [colormap(i) for i in np.linspace(0, 1, 60)]

        cbar_ax = fig.add_axes([0.2, 0.25, 0.7, 0.05])
        norm = colors.Normalize(vmin=0, vmax=60)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap),
                cax=cbar_ax, orientation='horizontal', label='Months')
    
    
        for idx, family in enumerate(families):
            with open(f"pkl_files/tsne_{family}.pkl",'rb') as f:
                tsne_result = pickle.load(f)
                
            for group in range(len(tsne_result)):
                if len(tsne_result[group]) > 0:
                    axs[idx].scatter(x=tsne_result[group][:,0], y=tsne_result[group][:,1], s=30, marker='o', color=colors_list[group])

            axs[idx].set_title(family)
            axs[idx].set_xlabel('')
            axs[idx].set_ylabel('')
            axs[idx].set_aspect('equal')
            axs[idx].set_xlim(maxlim)
            axs[idx].set_ylim(maxlim)
            if idx > 0:
                axs[idx].set_yticklabels('')
        fig.subplots_adjust(right=0.95, wspace=0.4,top=0.98)
        plt.tight_layout()
        save_output(fname)
        
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
    def __init__(self, results1, results2=None, label1='', label2=''):
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
        
        rc('font', **{'size': 40, 'family': 'serif', 'serif': ['Computer Modern Roman']})
        rc('text', usetex=True)
        rc('xtick', labelsize=40)
        rc('ytick', labelsize=40)
        rc('legend', fontsize=25)
        rcParams['figure.figsize'] = [16, 10]
    
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
                       label=f'F1 {self.label2} ', linewidth=2)
            alpha_1 = 0.3
        else:
            ax.plot(x, y1_recall, color= ColorMap['RECALL'].value, markersize=7, 
                    marker="^", label=f'Recall', linewidth=2)
            
            ax.plot(x, y1_prec, color= ColorMap['PREC'].value, markersize=7, 
                    marker="s", label=f'Precision', linewidth=2)

        # Plot f1 score for results 1
        ax.plot(x, y1_f1, color = ColorMap['F1'].value, markersize = 7, marker="o",
                label=f'F1 {self.label1} ', linewidth=2, alpha=alpha_1)
        ax.fill_between(x, 0, y1_f1, facecolor='none', hatch='/',
                        edgecolor='#BCDEFE', rasterized=True)

        months_of_x = len(self.results1['total_family'])+1
        labels_x = [str(x-6) if ((x > 6 and ((x-1) % 5 == 0 or x == 7)) or x == 1)
                    else '' for x in range(1,months_of_x)]

        ax.set_xticks(np.arange(-6,months_of_x-7))
        ax.set_xticklabels(labels_x)
        ax.set_ylabel("Performance")
        ax.legend(loc='best')
        ax.set_xlim(-6.5,months_of_x-7)
    
    def gradient_poi_selection(self,ax, k=3):
        """Function that selectionos POI by first calculating the difference of F1
        between 2 plots, then its change between X(t+1) and X(t). Plots the result
        along with markers for top 3 POI

        Args:
            ax (matplotlib.pyplot.axis): Matplotlib axis to plot on
        """        
        # Difference between F1 of results 2 and results 1
        difference = np.subtract(self.results2['f1'][1:],self.results1['f1'][1:])
        
        # Calculate difference of X(t+1) - X(t)
        gradient = np.diff(difference)
        
        # Insert 0 at the start to keep length consistence
        result = np.insert(gradient, 0,0)
        
        # Set points where results 1 out performs results 2 to 0, we only want improvements
        result = [n3 if n1 > n2 else 0 for n1,n2,n3 in zip(self.results2['f1'][1:],self.results1['f1'][1:], result)]
        result = np.array(result)

        # Select markers to plot
        markers_on = result.argsort(axis=0)[::-1][:k]
        print(markers_on)
        x = np.arange(len(result))
        
        # Plot result
        ax.plot(x, result,linewidth=2, alpha=0.6,label='Gradients',color='r')
        for n in markers_on:
            ax.axvline(n,linestyle='-.',color='r')
        ax.set_ylim(0,1)
      
    
    def distribution(self, ax, goodware=False):
        """Function that shows the distribution of families in the samples used
        for both training and testing. Training months are shifted to begin at -5 to 0 and
        testing months begin at 1. 
        TODO: Automate training shift

        Args:
            ax (matplotlib.pyplot.axis): Matplotlib axis to plot on
            goodware (bool, optional): Whether goodware should be shown. Defaults to False.
        """        
        families = self.get_families_used()
        groups = np.arange(-6, len(self.results1['total_family']) - 6)
        previous_count = np.array([0]* len(self.results1['total_family']))
        accepted_families = ["DOWGIN",'DNOTUA','KUGUO','AIRPUSH','REVMOB']
        
        others = np.array([0]* len(self.results1['total_family']))
        for family in families:
            if family in accepted_families:
                count_by_family = []
                for group in groups:
                    if family in self.results1['total_family'][group + 6]:
                        count_by_family.append(self.results1['total_family'][group + 6][family])
                    else:
                        count_by_family.append(0)

                ax.bar(groups, count_by_family, label=family, bottom=previous_count,
                    color=ColorMap[family].value)
                previous_count += np.array(count_by_family)
            else:
                count_by_family = []
                for group in groups:
                    if family in self.results1['total_family'][group + 6]:
                        count_by_family.append(self.results1['total_family'][group + 6][family])
                    else:
                        count_by_family.append(0)
                others += np.array(count_by_family)
                
        # Other families
        if len(families) > len(accepted_families):
            ax.bar(groups, others, label='OTHERS', bottom=previous_count, color='#800000')
            previous_count += np.array(others)
        
        if goodware:
            ax.bar(groups, previous_count, label='GOODWARE',bottom=previous_count, color=ColorMap['GOODWARE'].value)        
        
        months_of_x = len(self.results1['total_family'])+1
        labels_x = [str(x-6) if ((x > 6 and ((x-1) % 5 == 0 or x == 7)) or x == 1)
                    else '' for x in range(1,months_of_x)]
        
        ax.set_xticks(np.arange(-6,months_of_x-7))
        ax.set_xticklabels(labels_x)
        ax.set_ylabel('\# of samples')
        ax.legend(loc='best')
        ax.set_xlim(-6.5,months_of_x-7)
        plt.grid(visible=True, which='major', axis="y")

    
    def get_families_used(self, goodware=False):
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
        previous_count = [0] * len(self.results1['correct_family'])
        
        total_samples_all = [0] * len(self.results1['correct_family'])
        for family in familes:
            for group in range(len(self.results1['correct_family'])):
                if family in self.results1['total_family'][group + 6]:
                        total_samples_all[group] += self.results1['total_family'][group + 6][family]       
        # Loop through families
        for family in familes:
            results1_true_positives = []
            results2_true_positives = []
            
            # Get a count of individial families for every group
            for group in range(len(self.results1['correct_family'])):
                if family in self.results1['correct_family'][group]:
                    results1_true_positives.append(\
                        self.results1['correct_family'][group][family])
                else:
                    results1_true_positives.append(0)
                    
                if family in self.results2['correct_family'][group]:
                    results2_true_positives.append(\
                        self.results2['correct_family'][group][family])
                else:
                    results2_true_positives.append(0)
            
            # Find the difference between results2 and results 1 per family
            true_positive_diff = np.subtract(results2_true_positives,\
                                             results1_true_positives)
            # Normalise results to total number of samples per group  
            true_positive_diff_norm = true_positive_diff / total_samples_all
            
            # Check if month selection is given
            if month_selection != None:
                X = np.arange(len(month_selection))
                true_positive_diff_norm = true_positive_diff_norm[month_selection]
                if len(previous_count) != len(month_selection):
                    previous_count = [previous_count[x] for x in month_selection]
            else:
                X = np.arange(len(self.results1['correct_family']))
            
            # Remove negatives and NAN
            true_positive_diff_norm = [x if x >= 0 else 0 for x in true_positive_diff_norm]

            # Plot bar plot
            ax.bar(X, true_positive_diff_norm, color=ColorMap[family].value,\
                  label = family.capitalize(), bottom=previous_count)
            
            previous_count += np.array(true_positive_diff_norm)

        # Plot parameters
        ax.set_xticks(np.arange(len(X)))
        if month_selection == None:
            labels_x = [x if (x % 5 ==0 or x == 1) else '' for x in range(1,len(X)+1)]
        else:
            labels_x = np.add([1] * len(X), month_selection)
        ax.set_xticklabels(labels_x)
        ax.set_ylabel(f"Difference of true positives\nof \
                      {self.label1} and {self.label2} normalised")
        ax.set_ylim(0,1)
        ax.legend(loc='best')    
    
    def plot_performance_distribution(self, poi=True, fname='output.pdf'):
        """Function that creates performance plot on the top and distribution
        plot on the bottom
        
        Args:
            poi (bool, optional): Whether POI should be plotted on distribution graph
        """        
        fig, ax = plt.subplots(2,1)
        
        self.performance(ax[0])
        self.distribution(ax[1], True)      
                
        if self.results2 != None and poi == True:
            ax2 = ax[1].twinx()
            self.gradient_poi_selection(ax2, k=12)

            ax2.set_ylabel("Gradient of difference")
        
        ax[1].set_xlabel('Month')
        # plt.show()
        save_output(fname)
    
    def plot_single(self, plot, month_selection=None, fname='output.pdf'):
        """Function that creates one plot, takes in a plot type of 'distribution',
        'performance','difference'

        Args:
            plot (str): Type of plot to plot
            month_selection (list, optional): List of months to plot, if None then all months plotted.

        """        
        fig, ax = plt.subplots(1)
        months=None
        if month_selection != None:
            months = [x-1 for x in month_selection]
        if plot.lower() == 'distribution':
            self.distribution(ax)
        elif plot.lower() == 'performance':
            self.performance(ax)
        elif plot.lower() == 'difference':
            self.family_diff(ax, months)
        else:
            print("Plot type not found, try: \
                  distribution|performance|difference")
            return None
        
        plt.xlabel('Month')
        plt.grid(visible=True, which='major', axis="y")
        # plt.show()
        print(fname)
        save_output(fname)
        
        
class VizExpl():
    def __init__(self, results1, results2=None, label1='C1',label2='C2'):
        self.results1 = results1
        self.results2 = results2
        self.label1 = label1
        self.label2 = label2
    
    def get_top_feature_for_sample(self, md5_sample, k=5):
        """Given an md5 sample, get the top feature of it if its in
        the test set

        Args:
            md5_sample (str): MD5 sample to locate
            k (int, optional): Print top k features. Defaults to 5.
        """            
        # Find the md5 sample and save explanations
        for group_count, group in enumerate(self.results1['explanations'][1:]):
            for idx, md5 in enumerate(group[1]):
                if md5.lower() == md5_sample.lower():
                    explanation = group[0][idx]
                    group_number = group_count
                    break
    
        # Print md5, the family label and the prediction (0 for GW 1 for MW)
        print("MD5: {}".format(md5_sample.upper()))
        print(f"Group sample found in: {group_number + 1}")
        family_label = self.results1['family_class'][group_number][idx][1]
        pred_val = self.results1['family_class'][group_number][idx][0]
        print("Family Label: {}".format(family_label))
        print("Predicted as: {}".format(pred_val))
        
        # Get argsort of top k features
        topk_features = explanation.argsort(axis=0)[::-1][:k]
        print("-"*5)
        for idx, n in enumerate(topk_features):
            print(idx,self.results1['feature_names'][n], explanation[n])
    
    def feature_difference(self, group_selection, family_select=None, k=30):
        """Finds the difference between results2 topk features and results1 topk
        features.

        Args:
            group_selection (list): List of groups to return result.
            family_select (list, optional): List of families, if None then all families in test set.
            k (int, optional): Topk to select for each result. Defaults to 30.

        """        
        if self.results1 == None or self.results2 == None:
            print("Results 1 or Results 2 is None")
            return None 
        if family_select != None:
            family_select = list(map(str.upper,family_select))
        topk_features1 = self.get_top_features(self.results1, family_select=family_select, k=k)
        topk_features2 = self.get_top_features(self.results2, family_select=family_select, k=k)

        
        for group in group_selection:
            print(f'Group {group}: Top {k} features of {self.label2} - top {k} features of {self.label1}')
            results1_topk_set = set(topk_features1[group-1])
            results2_topk_set = set(topk_features2[group-1])
            
            new_features  = results2_topk_set - results1_topk_set
            
            # Print results
            print('-'*20)
            for idx, n in enumerate(new_features):
                print('{}   {}  {}'.format(idx, self.results1['feature_names'][n], n))     
    
    def get_samples_from_group(self,group_selection, family_selection, count=5):
        """For a given group selection, print out information about samples
        and their explanations

        Args:
            group_selection (list): List of groups to consider
            family_selection (list): List of families to filter
            count (int, optional): Number of samples to give. Defaults to 5.
        """        
        family_selection = list(map(str.upper,family_selection))
        running_count = 0
        for group in group_selection:
            print("-"*20)
            print(f"Group {group}")
            for idx, md5 in enumerate(self.results1['explanations'][1:][group-1][1]):
                if self.results1['family_class'][group-1][idx][1] in family_selection:
                    print("-"*5)
                    print(f"MD5: {md5}")
                    print(f"Family: {self.results1['family_class'][group-1][idx][1]}")
                    print(f"Predicted as: {self.results1['family_class'][group-1][idx][0]}")
                    
                    explanation = self.results1['explanations'][1:][group-1][0][idx]
                    topk_features = explanation.argsort(axis=0)[::-1][:5]
                    for x, n in enumerate(topk_features):
                        print(x,self.results1['feature_names'][n], explanation[n])
                
                    running_count += 1
                    if running_count >= count:
                        break
    
    def top_features_of_given_family(self, group_selection,family_selection, k=5):
        """Get the top features of a given family for given months. Prints out
        results in table format. This replicates table 5 & 6 of the paper.

        Args:
            result (list): Dictionary of results
            family_select (list): List of families to filter    

        Returns:
            numpy.ndarray: Array of indexes sorted
            numpy.ndarray: Raw explanation value unsorted

        """   
        filtered_list = self.get_explanations_mean_filter(self.results1, family_selection)
        topk, explanations = self.get_top_features_month(self.results1, filtered_list)
        feature_names = self.results1['feature_names']

        for family in family_selection:
            for group in group_selection:
                print('-'*10)
                print(f"Top {k} feature(s) for group {group} for family {family}")
                total = sum(abs(explanations[group-1]))
                
                for i, sample in enumerate(topk[group-1][:k]):
                    print(i, feature_names[sample], explanations[group-1][sample], round(explanations[group-1][sample]/total, 2) )
            


    def get_explanations_mean_filter(self, result, family_select, result_orig = []):
        """Filter out explanation samples for mean calculation, with the option of
        selecting missed samples only

        Args:
            result (list): Dictionary of results snooped
            family_select (list): List of samples to consider
            result_orig (list, optional): Dictionary of results unsnooped

        Returns:
            list: List of filtered samples
        """     
        filter_list = []
        if family_select != None:
            for group_num, group in enumerate(result['family_class']):
                filter_group = []
                for idx, sample in enumerate(group):
                    if sample[1] in family_select:
                        if result_orig == []:
                            filter_group.append(idx)
                        else:
                            # If select missed samples only
                            if sample[0] == 1 and result_orig['family_class'][group_num][idx][0] == 0:
                                filter_group.append(idx)

                filter_list.append(filter_group)
            
        return filter_list

    def get_explanations_mean_calculation(self, result, filter_list):
        """Calculate the mean of the explanations given a filter list

        Args:
            result (list): Dictionary of results snooped
            filter_list (list): List of samples to use for calculation

        Returns:
            list: List of meaned explanations
        """                        
        output = []
        # Take the mean of every sample
        for group_num, group in enumerate(result['explanations'][1:]):
            if isinstance(group[0], np.ndarray):
                if filter_list != []:
                    if filter_list[group_num] != []:
                        explanations = group[0][filter_list[group_num]]
                    else:
                        explanations = [np.zeros(1000), np.zeros(1000)]
                else:
                    explanations = group[0]
                output.append(np.mean(explanations, axis=0))
            else:
                output.append(0)
    
        return output
    
    def get_top_features(self, result, filter_list):
        """Get the mean of weights of all samples from selected family to produce
        a global explanation

        Args:
            result (list): Dictionary of results
            family_select (list): List of families to filter    

        Returns:
            numpy.ndarray: Array of indexes sorted
            numpy.ndarray: Raw explanation value unsorted

        """   
        explanation_per_month = self.get_explanations_mean_calculation(result, filter_list)
        global_explanation = np.mean(explanation_per_month, axis=0)
        sorted_global_explanation = np.argsort(global_explanation)[::-1]

        return sorted_global_explanation, global_explanation
    
    def get_top_features_month(self, result, filter_list):
        """Get the mean of weights of all samples from selected family to produce
        a global explanation. This is only done for each month, hence returned
        array is split in to multiple months

        Args:
            result (list): Dictionary of results
            family_select (list): List of families to filter    

        Returns:
            numpy.ndarray: Array of indexes sorted
            numpy.ndarray: Raw explanation value unsorted

        """   
        explanation_per_month = self.get_explanations_mean_calculation(result, filter_list)
        sorted_global_explanation = []
        for explanations in explanation_per_month:
            sorted_global_explanation.append(np.argsort(explanations)[::-1])

        return sorted_global_explanation, explanation_per_month


    def mean_of_weights_of_top_feature_of_missed_family_samples(self, family_selection, k=5, missed=False):
        """Get the mean of weights of top features of all samples from selected family. Prints
        result as a table. This replicates table 1 and 3 of the paper.

        Args:
            family_select (list): List of families to filter            
            k (int, optional): Number of top features to show. Defaults to 5.

        """        
        if missed:
            filtered_list = self.get_explanations_mean_filter(self.results2, family_selection,self.results1)
        else:
            filtered_list = self.get_explanations_mean_filter(self.results2, family_selection)

        _, values1 = self.get_top_features(self.results1, filtered_list)
        topk_features2, values2 = self.get_top_features(self.results2, filtered_list)

        # Print result as table 
        print("Top 5 features of C2 and their value changes from C1 to C2")
        print('-'*10)
        for n in range(5):
            print(self.results1['feature_names'][topk_features2][n], round(values1[topk_features2][n]*100,3), \
                  round(values2[topk_features2][n]*100,3), str(round((values2[topk_features2][n]/values1[topk_features2][n]*100),3)) + "%")

        
class FamilyIso(Viz):
    def __init__(self, results1, results2, results3, results4, results5):
        self.results = [results1, results2, results3, results4, results5]
        rc('font', **{'size': 20, 'family': 'serif', 'serif': ['Computer Modern Roman']})
        rc('text', usetex=True)
        rc('xtick', labelsize=6)
        rc('ytick', labelsize=7)
        
    def plot_family_iso_matrix(self, fname='output.pdf'):
        """Plots recall of solo train experiments and prints output as table. This replicates 
        table 7 of the paper.
        """     

        fig, ax = plt.subplots(5,6, sharex=True, sharey=True)
        
        testing_families = ['DOWGIN','DNOTUA','KUGUO','AIRPUSH','REVMOB','GOODWARE']
        # Get all training familes
        training_families = []
        for n in range(len(self.results)):
            # if n == 1:
            #     training_families.append(self.results[n]['test_amount'][0][0])
            # else:
                training_families.append(self.results[n]['test_amount'][0][0][0])

        # Get total number of samples
        grand_total = []
        for month in self.results[0]['total_family'][6:]:
            grand_total.append(sum(month.values()))
                
        for x, training_family in enumerate(training_families):
            total = self.results[x]['total_family']
            correct = self.results[x]['correct_family']
            output = []
            for y, testing_family in enumerate(testing_families):
                total_family = []
                correct_family = []
                for month in total:
                    if testing_family in month:
                        total_family.append(month[testing_family])
                    else:
                        total_family.append(0)
                for month in correct:
                    if testing_family in month:
                        correct_family.append(month[testing_family])
                    else:
                        correct_family.append(0)
                
                total_family_normalised = np.divide(total_family[6:],grand_total)
                correct_family_normalised = np.divide(correct_family,grand_total)

                np.nan_to_num(correct_family_normalised, copy=False,nan=0.0)
                ax[x, y].plot(np.arange(len(total_family_normalised)),total_family_normalised, color='black', linewidth=0.2)
                ax[x, y].fill_between(np.arange(len(correct_family_normalised)),0,correct_family_normalised,facecolor=ColorMap[testing_family].value)
                # ax[x, y].plot(np.arange(len(correct_family_normalised)),correct_family_normalised,color=ColorMap[testing_family].value)
                output.append(round((sum(correct_family)/sum(total_family[6:]))*100,2))
                
                if x == 0:
                    ax[x,y].set_title(testing_family, fontsize=10)
                if y == 0:
                    ax[x,y].set_ylabel(training_family, fontsize=10)

                if y == len(training_families)-1:
                    ax[x,y].set_xticks(np.arange(0,54,5))
                    labels_x = [0,5,10,15,20,25,30,35,40,45,50]
                    ax[x,y].set_xticklabels(labels_x)
                ax[x,y].set_xticklabels(ax[x,y].get_xticks(), rotation=-90)
                    
            print(output)
        
        plt.subplots_adjust(hspace = 0.05, wspace = 0.05)
        # plt.show()
        save_output(fname)
