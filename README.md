# Dataset Analysis

Repository containing code for our [DIMVA paper](https://www.overleaf.com/project/636e75bbd657c38e61acf45f).
Link to dataset can be found [Here](https://emckclac-my.sharepoint.com/:u:/g/personal/k21129040_kcl_ac_uk/ERcKWdTr_bxFmR5fIE2xgqYBn_LrTom8IfmnwgWyOGVA-w?e=nIAMn9)
## Getting Started

### Installation
This project requires Python 3 as well as the statistical learning stack of NumPy, SciPy and Scikit-learn, secml.

First, install package dependencies using the listing in `requirements.txt`. 

```
pip install -r requirements.txt
```

## Usage
The `main.py` script contains an example of running experiments using the transcedent dataset. Currently there are three experiments available: 'base', 'half' and 'snoop'. Below are steps on running a specific experiment.

### Load dataset
First load in the desired dataset and obtain the X predictors *X*, y predictors *y*, timestamps *t*, family labels *f*, feature names *feature_names* and md5 *md5*. 
```python
PATH = "../Datasets/extended-features/"
X, y, t, f, feature_names, md5 = load_transcend(f"{PATH}extended-features-X-updated.json",
                                                f"{PATH}extended-features-y-updated.json",
                                                f"{PATH}extended-features-meta-updated.json",
                                                f"{PATH}meta_info_file.tsv")
```

### Reduce feature space
Reduce the feature space to a manageable amount and save the feature indexes as a pkl file
```python
    X, feature_names = util.feature_reduction(X, y, feature_names, "pkl_files/feature_index_1000.pkl", feature_size=1000)
```


### Dataset class
Put the data in to a dataset class, this gives us flexibiliy when selecting samples. Currently there are 2 main functions in the dataset class, splitting the dataset in to time aware splits for analysis and finding occurences of features in the dataset. 
```python
    dataset = Dataset(X, y, t, f, feature_names, md5)
```

Search up feature name IDs
```python
    ids = dataset.get_feature_id_from_name("android")
```

Find IDs in family
```python
    dataset.sample_select_from_feature_id(families=['Dowgin','Dnotua','Kuguo','Airpush','Revmob'],ids=ids,contains=True, year=2015, month=1)
```

Split dataset and return time aware indexes for training and test
```python
    train, test = dataset.time_aware_split_index('month', 6, 1)
```

### Analysis
The analysis class runs the experiment outlined in the paper. Currently there are 3 main experiments, base, half and snoop. The results of this will be logged in a MySQL database and the results in a pkl file. By default, a file name ```pkl_files``` needs to be created.
```python
    analyse = Analysis(X, y, t, f, feature_names, train, test)

    training_family = ['Dowgin','Dnotua','Kuguo','Airpush','Revmob']
    testing_family = ['Dowgin','Dnotua','Kuguo','Airpush','Revmob']

    analyse.run(training_family=training_family, testing_family=testing_family,experiment='snoop', dataset='Transcend')
```

### Visualizing data
To visualise the results, we first load in the corresponding data in question. ```The ResultsLoader()``` class gives an easy way in accessing saved experiments.

```python
    training_familes = ['Dowgin','Dnotua','Kuguo','Airpush','Revmob']
    testing_families = ['Dowgin','Dnotua','Kuguo','Airpush','Revmob']
    ResultsLoader().query_database_for_ID('half',training_familes,testing_families,'Transcend')
```

Load in the desired data using the ID returned by ```ResultsLoader()```
```python
    result1 = ResultsLoader().load_file_from_id(5)
    result2 = ResultsLoader().load_file_from_id(6)
```

For performance, distribution and difference plots
```python
    Viz(result1,result2).plot_performance_distribution()
    Viz(result1,result2).plot_single('difference')
```

For explantion related plots 
```python
    VizExpl(results1).get_explanations_mean(base)
```