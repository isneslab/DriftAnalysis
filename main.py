from dataset import Dataset
from load import load_transcend
from analysis import Analysis
import util


# Load in Transcend
PATH = "../Datasets/extended-features/"
X, y, t, f, feature_names, md5 = load_transcend(f"{PATH}extended-features-X-updated.json",
                                                f"{PATH}extended-features-y-updated.json",
                                                f"{PATH}extended-features-meta-updated.json",
                                                f"{PATH}meta_info_file.tsv")


# Reduce feature space
# X, feature_names = util.feature_reduction(X, y, feature_names, "../dataset-analysis/Multi Family Temporal Analysis/pkl_files/feature_index_1000.pkl", feature_size=1000)
# X, feature_names = util.feature_reduction(X, y, feature_names, "pkl_files/feature_index_1000.pkl", feature_size=1000)


# Put data in to dataset class
dataset = Dataset(X, y, t, f, feature_names, md5)

# Search up feature name ID and get samples
# ids = dataset.get_feature_id_from_name("google-analytics")
# ids = [630]
# dataset.sample_select_from_feature_id(families=['Dnotua'],ids=ids,contains=True,month=4, year=2017)

# Split dataset and return time aware indexes for training and test
train, test = dataset.time_aware_split_index('month', 6, 1)




# Run analysis
analyse = Analysis(X, y, t, f, feature_names, md5, train, test)

# # training_family = ["Dowgin",'Dnotua','Kuguo','Airpush','Revmob', 'Smsreg','Leadbolt','Inmobi','Anydown','Feiwo','Gappusin','Ewind','Baiduprotect','Hiddad','Mecor','Zdtad','Dasu','Mobidash','Viser','Autoins']
# # testing_family = ["Dowgin",'Dnotua','Kuguo','Airpush','Revmob', 'Smsreg','Leadbolt','Inmobi','Anydown','Feiwo','Gappusin','Ewind','Baiduprotect','Hiddad','Mecor','Zdtad','Dasu','Mobidash','Viser','Autoins']

training_family = ['Dowgin','Dnotua','Kuguo','Airpush','Revmob']
testing_family = ['Dowgin','Dnotua','Kuguo','Airpush','Revmob']
analyse.run(training_family=training_family, testing_family=testing_family,experiment='half_random', dataset='Transcend')
# analyse.run(training_family=None, testing_family=None,experiment='base', dataset='Transcend')


