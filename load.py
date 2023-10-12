"""
load.py
~~~~~~~~~~~

This module loads in data from a given file and adds
functionality to manipulate them to the required
format for other modules. Each dataset requires its own loading function.

"""
import datetime
import numpy as np
import json
import os
import pandas as pd
from tqdm import tqdm
import pickle
import util

from sklearn.feature_extraction import DictVectorizer

def load_file(filename):
    """Helper function to load json files
    
    Arg:
        filename (str): File path of json file to be loaded

    Returns:
        (json object): json file object 

    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return data

def get_family_name(labels):
    """Helper function for getting family names"""
    for label in labels:
        if label.startswith('FAM'):
            family = label.split('FAM:')[1].split('|')[0]
            return family


def load_transcend(X, y, meta_info, meta_family):
    """Loading function for transcend dataset

    Args:
        X (str): File path of X
        y (str): File path of y
        meta_info (str): File path of meta info file containing timestamps and md5
        meta_family (str): File path of tsv file containing family labels

    Returns:
        (scipy.sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
            Array of predictors X
            Array of predictors y
            Array of time stamps
            Array of family labels
            Array of feature names
            Array of md5
    """        
    print("Loading Transcend dataset, this can take up to 3 minutes...")

    # Load in X and convert to numpy ndarray
    X = load_file(X)
    vec = DictVectorizer()
    X = vec.fit_transform(X).astype("float32")
    feature_names = vec.get_feature_names_out()
    print("X loaded")

    # Load in y
    y = load_file(y)
    y = np.asarray(y).flatten()
    print("y loaded")

    # Load in time
    meta_file = load_file(meta_info)
    t = [n['dex_date'] for n in meta_file]
    t = [datetime.datetime.strptime(n, '%Y-%m-%dT%H:%M:%S') if "T" in n
             else datetime.datetime.strptime(n, '%Y-%m-%d %H:%M:%S') for n in t]
    t = np.array(t)
    print("Timestamps loaded")

    # Get md5 for each sample
    md5 = [n['md5'].upper() for n in meta_file]
    
    # Read family meta info file as dataframe
    print("Loading family labels")
    family_pd = pd.read_csv(meta_family, delimiter='\t')

    # Splice out samples with family labels
    family_pd = family_pd.loc[~family_pd['families'].isnull() & ~(family_pd['families'] == '[]')]

    # Find family label from md5
    if os.path.exists("pkl_files/family_labels.pkl"):
        with open("pkl_files/family_labels.pkl","rb") as file:
            data = pickle.load(file)
        index_with_families = data[0]
        f = data[1]
    
    else:
        index_with_families = []
        f = []

        family_pd['md5'] = family_pd['md5'].str.upper()
        malware_with_labels = family_pd.loc[family_pd["md5"].isin(md5)]

        for idx, md5_sample in tqdm(enumerate(md5), total=len(md5)):
            if y[idx] == 0:
                f.append('GOODWARE')
                index_with_families.append(idx)
            elif md5_sample in list(malware_with_labels['md5']):
                labels = malware_with_labels['families'].loc[malware_with_labels['md5'] == md5_sample].to_list()
                labels = labels[-1].split(',')

                if get_family_name(labels):
                    family = get_family_name(labels).upper()
                    f.append(family.upper())
                    index_with_families.append(idx)
            else:
                pass
        with open("pkl_files/family_labels.pkl","wb") as file:
            pickle.dump([index_with_families,f],file)

    print("Family labels loaded")

    # Feature reduction
    X, feature_names = util.feature_reduction(X, y, feature_names, "pkl_files/feature_index_1000_before_greyware.pkl", feature_size=1000)

    y = y[index_with_families]
    X = X[index_with_families]
    t = t[index_with_families]
    f = np.array(f)
    md5 = np.array(md5)[index_with_families]
    print("Finished loading Transcend dataset")
    return X, y, t, f, feature_names, md5
    
