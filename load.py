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
 
md5_counter_int = 1


def load_npz_files(filename):
    folder = '../Datasets/data/gen_apigraph_drebin'
    saved_data_file = os.path.join(folder, filename)
    data = np.load(saved_data_file)
    X_train, y_train = data['X_train'], data['y_train']
    y_mal_family = data['y_mal_family']
    ben_len = X_train.shape[0] - y_mal_family.shape[0]
    y_ben_family = np.full(ben_len, 'GOODWARE')
    all_train_family = np.concatenate((y_mal_family, y_ben_family), axis=0).upper()
    y_train_new = [y if y == 0 else 1 for y in y_train]
    return X_train, y_train_new, all_train_family
    
def generate_fake_md5():
    global md5_counter_int
    new_fake_md5 = str(md5_counter_int).zfill(10)
    return new_fake_md5
    

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

def load_pkl_file(filename):
    """Helper function to load pkl files
    
    Arg:
        filename (str): File path of pkl file to be loaded

    Returns:
        (json object): pkl file object 

    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
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
    # y = np.asarray(y).flatten()
    y = np.asarray(y).flatten()
    y = y.astype(int)
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
    # X, feature_names = util.feature_reduction(X, y, feature_names, "pkl_files/feature_index_1000_before_greyware.pkl", feature_size=1000)

    y = y[index_with_families]
    X = X[index_with_families].toarray()
    t = t[index_with_families]
    f = np.array(f)
    md5 = np.array(md5)[index_with_families]
    print("Finished loading Transcend dataset")
    return X, y, t, f, feature_names, md5
    
def load_apigraph_drebin():
    """Loading function for Apigraph drebin dataset
    Returns:
        (scipy.sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
            Array of predictors X
            Array of predictors y
            Array of time stamps
            Array of family labels
            Array of feature names
            Array of md5
    """        
    print("Loading APIGgraph drebin dataset, this can take up to 3 minutes...")

    
    # Load in 1 year of training data
    X, y, f = load_npz_files('2012-01to2012-12_selected.npz')
    
    t = []
    md5 = []
    for n in range(len(y)):
        month = 1
        md5.append(generate_fake_md5())
        t.append(datetime.datetime(2012,1,1,1,1,1))
        if n > int(len(y)/12) * 1:
            month += 1
    
    feature_names = load_file('2012-01to2012-12_selected_training_features.json')    
    for year in [2013,2014,2015,2016,2017,2018]:
        for month in range(1,13):
            if year == 2018 and month > 10:
                break
            month = str(month).zfill(2)
            filename = f'{year}-{month}_selected.npz'
            X_test, y_test, f_test = load_npz_files(filename)
            t_test = []
            md5_test = []
            for n in range(len(y_test)):
                md5.append(generate_fake_md5())
                t.append(datetime.datetime(year,month,1,1,1,1))

            X = np.concatenate((X, X_test))
            y = np.concatenate((y, y_test))
            f = np.concatenate((f, f_test))
            t = np.concatenate((t, t_test))
            md5 = np.concatenate((md5, md5_test))
 
    print("Finished loading apigraph drebin dataset")
    
    f = np.array(f)
    md5 = np.array(md5)
    
    return X, y, t, f, feature_names, md5
    
