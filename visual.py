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
        
        rc('font', **{'size': 40, 'family': 'serif', 'serif': ['Computer Modern Roman']})
        rc('text', usetex=True)
        rc('xtick', labelsize=40)
        rc('ytick', labelsize=40)
        rc('legend', fontsize=30)
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
                       label=f'F1', linewidth=2)
            alpha_1 = 0.3
        else:
            ax.plot(x, y1_recall, color= ColorMap['RECALL'].value, markersize=7, 
                    marker="^", label=f'Recall', linewidth=2)
            
            ax.plot(x, y1_prec, color= ColorMap['PREC'].value, markersize=7, 
                    marker="s", label=f'Precision', linewidth=2)

        # Plot f1 score for results 1
        ax.plot(x, y1_f1, color = ColorMap['F1'].value, markersize = 7, marker="o",
                label=f'F1 ', linewidth=2, alpha=alpha_1)
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
    
    def gradient_poi_selection(self,ax):
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
        markers_on = result.argsort(axis=0)[::-1][:3]
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
        self.distribution(ax[1], True)      
                
        if self.results2 != None and poi == True:
            ax2 = ax[1].twinx()
            self.gradient_poi_selection(ax2)

            ax2.set_ylabel("Gradient of difference")
        
        ax[1].set_xlabel('Month')
        plt.show()
    
    def plot_single(self, plot, month_selection=None):
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
        plt.show()
        
        
class VizExpl():
    def __init__(self, results1, results2=None, label1='C1',label2='C2'):
        self.results1 = results1
        self.results2 = results2
        self.label1 = label1
        self.label2 = label2

        
    def get_explanations_mean(self, result, family_select):
        """Calculate the mean of the explanations

        Args:
            result (list): Dictionary of results
            family_select (list): List of samples to consider

        Returns:
            list: List of meaned explanationed
        """        
        # If family select, compute a filter list
        filter_list = []
        if family_select != None:
            for group in result['family_class']:
                filter_group = []
                for idx, sample in enumerate(group):
                    if sample[1] in family_select:
                        filter_group.append(idx)
            
                filter_list.append(filter_group)
                
        output = []
        # Take the mean of every sample
        for group_num, group in enumerate(result['explanations'][1:]):
            if isinstance(group[0], np.ndarray):
                if filter_list != []:
                    if filter_list[group_num] != []:
                        explanations = group[0][filter_list[group_num]]
                    else:
                        output.append(0)
                else:
                    explanations = group[0]
                output.append(np.mean(explanations, axis=0))
            else:
                output.append(0)
    
        return output
    
    def get_top_features(self, result, family_select=None, k=5):
        """Get the top features of each month and returns a filter for them

        Args:
            result (dict): Dictionary of result
            family_select (list, optional): List of families to filter, if None then all families.
            k (int, optional): Number of top features to show. Defaults to 5.

        Returns:
            list: List of filters for top k
        """        
        
        expl_mean = self.get_explanations_mean(result, family_select)
        output = []
        
        for group in expl_mean:
            if isinstance(group, np.ndarray):
                topk_filter = group.argsort(axis=0)[::-1][:k]
                output.append(topk_filter)
            else:
                output.append('None')
        
        return output
    
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
                
class FamilyIso(Viz):
    def __init__(self, results1, results2, results3, results4, results5):
        self.results = [results1, results2, results3, results4, results5]
        
    def plot_family_iso_matrix(self):
        
        fig, ax = plt.subplots(5,6, sharex=True, sharey=True)
        
        testing_families = ['DOWGIN','DNOTUA','KUGUO','AIRPUSH','REVMOB','GOODWARE']
        
        # Get all training familes
        training_families = []
        for n in range(len(self.results)):
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

                ax[x, y].plot(np.arange(len(total_family_normalised)),total_family_normalised, color='black')
                ax[x, y].fill_between(np.arange(len(correct_family_normalised)),0,correct_family_normalised,facecolor=ColorMap[testing_family].value)
                
                output.append(round((sum(correct_family_normalised)/sum(total_family_normalised))*100,2))
                
                if x == 0:
                    ax[x,y].set_title(testing_family, fontsize=25)
                if y == 0:
                    ax[x,y].set_ylabel(training_family, fontsize=20)

                if y == len(training_familes)-1:
                    labels_x = [a if (a % 5 ==0 or a == 1) else '' for a in range(1,55)]
                    ax[x,y].set_xticks(np.arange(54))
                    ax[x,y].set_xticklabels(labels_x)
                    
            print(output)
                    
        plt.subplots_adjust(hspace = 0.05, wspace = 0.05)
        plt.show()
            
        
        
    
if __name__=='__main__':
    print("Visual module")
    training_familes = ['Dowgin','Dnotua','Kuguo','Airpush','Revmob']
    testing_families = ['Dowgin','Dnotua','Kuguo','Airpush','Revmob']
    # ResultsLoader().query_database_for_ID('snoop',training_familes,testing_families,'Transcend')
    # c1 = ResultsLoader().load_file_from_id(14) # half
    # c2 = ResultsLoader().load_file_from_id(15) # snoop
    # c3 = ResultsLoader().load_file_from_id(17) # snoop no gw
    # all = ResultsLoader().load_file_from_id(18) # all families
    # dnotua_all = ResultsLoader().load_file_from_id(19) # trained on all families + dnotua month 31
    # all_ish = ResultsLoader().load_file_from_id(21) # all families (actually just 20)
    # dnotua_all_ish = ResultsLoader().load_file_from_id(39) # all families (actually just 20) + dnotua month 31    
    # c1_random = ResultsLoader().load_file_from_id(31) # half random + fixed incorrect sampling
    # c2_random = ResultsLoader().load_file_from_id(32) # snoop random + fixed incorrect sampling
    # c3_random = ResultsLoader().load_file_from_id(34) # snoop no gw random + fixed incorrect sampling
    # all = ResultsLoader().load_file_from_id(37) # All families
    # dnotua_all = ResultsLoader().load_file_from_id(38) # All familes + dnotua training
    # dnotua4_all = ResultsLoader().load_file_from_id(39) # All families + dnotua 4 training
    # dnotua3_all = ResultsLoader().load_file_from_id(41) # All families + dnotua 3 training
    
    dowgin_train = ResultsLoader().load_file_from_id(44) # Train on dowgin
    dnotua_train = ResultsLoader().load_file_from_id(48) # Train on dnotua
    kuguo_train = ResultsLoader().load_file_from_id(45) # Train on kuguo
    airpush_train = ResultsLoader().load_file_from_id(46) # Train on airpush
    revmob_train = ResultsLoader().load_file_from_id(47) # Train on revmob
    
    
    
    
    # visual = Viz(dowgin_train)
    # visual.plot_performance_distribution()
    # visual.plot_single('performance')
    
    # VizExpl(c3,c2).feature_difference(group_selection=[40, 45, 50])
    # VizExpl(c2).get_top_feature_for_sample('b073f248d7817bced11e07bb4fcb5c43')
    # VizExpl(c1).get_samples_from_group(group_selection=[46],family_selection=['Dnotua'])
    
    FamilyIso(dowgin_train,dnotua_train,kuguo_train,airpush_train,revmob_train).plot_family_iso_matrix()