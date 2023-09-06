"""
dataset.py
~~~~~~~~~~~

A module for storing and acessing temporal data for a multi family
dataset. The main functionality of this module is to store temporal
data such that it is easily accessable.

"""

import bisect
import operator

from tqdm import tqdm
import numpy as np
from dateutil.relativedelta import relativedelta
from collections import Counter
import calendar

class Dataset():
    def __init__(self, X, y, t, f, feature_names, md5=None):
        """Class that handles all time aware splits and provides
        functionality for sampling from feature names

        Args:
            X (np.ndarray): Multi-dimensional array of predictors
            y (np.ndarray): Array of binary output labels
            t (np.ndarray): Array of timestamp tags
            f (np.ndarray): Array of family labels
            feature_names (np.ndarray): Array of feature names
            md5 (np.ndarray, optional): Array of md5 for sampling.If None,
            sampling functions will not work.
        """        
        self.X = X
        self.y = y
        self.t = t
        self.f = f
        self.feature_names = feature_names
        self.md5 = md5


    def time_aware_split_index(self, granularity, train_windows, test_window):
        """Function that partitions list t by time

        Args:
            granularity (str): The unit of time used to denote the window size.
            Acceptable values are 'year|quater|month|week|day'.

        Returns:
            (list, list):
                Indexing for the training partition
                List of indexings for the testing partitions

        """

        # Order the dates as well as their original index positions
        with_indexes = zip(self.t, range(len(self.t)))
        time_sorted_with_index = sorted(with_indexes, key=operator.itemgetter(0))

        # Split out he dates from the indexes
        dates = [tup[0] for tup in time_sorted_with_index]
        indexes = [tup[1] for tup in time_sorted_with_index]

        # Get earliest start date
        
        trains = []
        from_idx = 0
        for _ in range(train_windows):   
            # Slice out training partition
            relative_date = dates[from_idx] + get_relative_delta(1, granularity)

            # Calculate last day of the month
            boundary = relative_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            
            to_idx = bisect.bisect_left(dates, boundary)
            trains.append(indexes[from_idx:to_idx])
            from_idx = to_idx       
        
        tests = []
        while to_idx < len(indexes):
            relative_date = dates[to_idx] + get_relative_delta(test_window, granularity)
            boundary = relative_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            from_idx = to_idx
            to_idx = bisect.bisect_left(dates, boundary, to_idx)
            tests.append(indexes[from_idx:to_idx])

        new_tests = []
        # Attempt to recreate starting state here
        for test in tests:
            new_test = sorted(test)
            new_tests.append(new_test)
            
        new_trains = []
        for train in trains:
            new_train = sorted(train)
            new_trains.append(new_train)
            
        return new_trains, new_tests
    
    
    def family_selection(self, families):
        """Function  that selects families from a list 
        of family labels and return the indexes

        Args:
            families (np.ndarray): Array of family labels

        Returns:
            list: List of indexes of selected families
        """

        output_index = []
        for idx, sample in tqdm(enumerate(self.f)):
            if sample in families:
                output_index.append(idx)
        
        return output_index

    def get_feature_id_from_name(self, feature_name):
        """Find feature ID from a given string

        Args:
            feature_name (str): String to search features by
        """        
        string_array = np.char.lower(self.feature_names.astype('str'))
        target_string = feature_name.lower()
        
        indices = np.where(np.char.find(string_array, target_string) >= 0)[0]
        
        print("Found {} features that contain string".format(len(indices)))
        
        for count, index in enumerate(indices):
            print(f"{count}   {self.feature_names[index]}   {index}")
        
        return indices
        
    def get_date_from_md5(self, md5):
        for idx, sample in enumerate(self.md5):
            if sample.upper() == md5.upper():
                print(self.t[idx].year)
                print(self.t[idx].month)
                print(self.f[idx])
    
    
    def sample_select_from_feature_id(self,families, ids, contains=True,year=None,month=None, md5_samples=3):
        """Function that selects samples of a given feature ID

        Args:
            families (list): List of families to check
            ids (list): List of feature IDs to check
            contains (bool, optional): If True, samples of md5 will contain the feature
            year (int, optional): Year of data to check. If None, all data is checked
            month (int, optional): Month of data to check. Defaults to None.
            md5_samples (int, optional): Number of md5 samples printed out
        """               
        # Decalare variables
        X = self.X
        f = self.f
        t = self.t
        md5 = self.md5
            
        # If year and/or month given, select those years and months
        time_index_filter = []
        if year != None:
            for idx in range(len(self.t)):
                if self.t[idx].year == year:
                    if month == None:
                        time_index_filter.append(idx)
                    elif self.t[idx].month == month:
                        time_index_filter.append(idx)
                    else:
                        continue
                    
            X = self.X[time_index_filter]
            f = self.f[time_index_filter]
            t = self.t[time_index_filter]
            md5 = self.md5[time_index_filter]
        
        # Search for samples with given feature IDs
        total = Counter(f)
        for id in ids:
            selected_feature_X = X[:,id]
            selected_feature_X = [i > 0 for i in selected_feature_X]
            families_with_feature = f[selected_feature_X]
            output = Counter(families_with_feature)
            
            # Print results   
            print("-"*20)
            print("Feature {}: {}".format(id, self.feature_names[id]))            
            for family in families:
                print("-"*10)
                print("{} {}/{}".format(family, output[family.upper()], total[family.upper()]))
                # Check if we should print samples that contain that feature or does NOT contain
                if not contains:
                    selected_feature_X = [not n for n in selected_feature_X]
                # Pick out samples of that family
                family_selected_filter = []
                for idx in range(len(selected_feature_X)):
                    if selected_feature_X[idx] == True and f[idx] == family.upper():
                        family_selected_filter.append(True)
                    else:
                        family_selected_filter.append(False)
                # Print MD5 samples          
                print("MD5 samples")
                for n in range(md5_samples):
                    try:
                        print("{} {} {}".format(n,md5[family_selected_filter][n-1], t[family_selected_filter][n-1]))
                    except:
                        pass


def get_relative_delta(offset, granularity):
    """Get delta of size 'granularity'.

    Args:
        offset: The number of time units to offset by.
        granularity: The unit of time to offset by, expects one of
            'year', 'quarter', 'month', 'week', 'day'.

    Returns:
        The timedelta equivalent to offset * granularity.

    """
    # Make allowances for year(s), quarter(s), month(s), week(s), day(s)
    granularity = granularity[:-1] if granularity[-1] == 's' else granularity
    try:
        return {
            'year': relativedelta(years=offset),
            'quarter': relativedelta(months=3 * offset),
            'month': relativedelta(months=offset),
            'week': relativedelta(weeks=offset),
            'day': relativedelta(days=offset),
        }[granularity]
    except KeyError:
        raise ValueError('granularity not recognised, try: '
                         'year|quarter|month|week|day')