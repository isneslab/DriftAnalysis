import os
from dataset import Dataset
from load import load_transcend
from analysis import Analysis
import visual

# Create dirs to store results
if not os.path.isdir('./pkl_files'):
    os.mkdir('pkl_files')

if not os.path.isdir('./results'):
    os.mkdir('results')

# Load in Transcend
PATH = "../Datasets/extended-features/"
X, y, t, f, feature_names, md5 = load_transcend(f"{PATH}extended-features-X-updated.json",
                                                f"{PATH}extended-features-y-updated.json",
                                                f"{PATH}extended-features-meta-updated.json",
                                                f"{PATH}meta_info_file.tsv")


# Put data in to dataset class
dataset = Dataset(X, y, t, f, feature_names, md5)

# Split dataset and return time aware indexes for training and test
train, test = dataset.time_aware_split_index('month', 6, 1)

# Run Tsne experiments
Analysis(X, y, t, f, feature_names, md5, train, test).tsne(['Dowgin','Dnotua','Kuguo','Airpush','Revmob'])

# Train and test for all families
all_families_id = Analysis(X, y, t, f, feature_names, md5, train, test).run(
                training_family=None, testing_family=None,experiment='base', dataset='Transcend') # All families base
all_families_dnotua_id = Analysis(X, y, t, f, feature_names, md5, train, test).run(
                training_family=None, testing_family=None,experiment='base_additional', dataset='Transcend') # All families with dnotua

# Select families
training_family = ['Dowgin','Dnotua','Kuguo','Airpush','Revmob']
testing_family = ['Dowgin','Dnotua','Kuguo','Airpush','Revmob']

# Run analysis
cbase_id = Analysis(X, y, t, f, feature_names, md5, train, test).run(
                training_family=training_family, testing_family=testing_family,experiment='half_random', dataset='Transcend') # C base experiment
c1_id = Analysis(X, y, t, f, feature_names, md5, train, test).run(
                training_family=training_family, testing_family=testing_family,experiment='snoop_random', dataset='Transcend') # C 1 experiment
c2_id = Analysis(X, y, t, f, feature_names, md5, train, test).run(
                training_family=training_family, testing_family=testing_family,experiment='nogwsnoop_random', dataset='Transcend') # C 2 experiment

# Loops for solo training
loop_id = []
for train_fam in training_family:
    if train_fam == 'Dnotua':
        experiment_t = 'base_additional'
    else:
        experiment_t = 'base'
    id = Analysis(X, y, t, f, feature_names, md5, train, test).run(
                training_family=[train_fam], testing_family=testing_family,experiment=experiment_t, dataset='Transcend') # Solo family experiments
    loop_id.append(id)

# Write IDs in file for future reference
with open("results/results_files_id",'w') as f:
    f.writelines(f"all_families_id: {all_families_id}")
    f.writelines(f"all_families_dnotua_id: {all_families_dnotua_id}")
    f.writelines(f"cbase_id: {cbase_id}")
    f.writelines(f"c1_id: {c1_id}")
    f.writelines(f"c2_id: {c2_id}")
    f.writelines(f"dowgin_solo_id: {loop_id[0]}")
    f.writelines(f"dnotua_solo_id: {loop_id[1]}")
    f.writelines(f"kuguo_solo_id: {loop_id[2]}")
    f.writelines(f"airpush_solo_id: {loop_id[3]}")
    f.writelines(f"revmob_solo_id: {loop_id[4]}")

# Load in files
all_families = visual.ResultsLoader().load_file_from_id(all_families_id)
all_families_dnotua = visual.ResultsLoader().load_file_from_id(all_families_dnotua_id)
cbase = visual.ResultsLoader().load_file_from_id(cbase_id)
c1 = visual.ResultsLoader().load_file_from_id(c1_id)
c2 = visual.ResultsLoader().load_file_from_id(c2_id)
dowgin_solo = visual.ResultsLoader().load_file_from_id(loop_id[0])
dnotua_solo = visual.ResultsLoader().load_file_from_id(loop_id[1])
kuguo_solo = visual.ResultsLoader().load_file_from_id(loop_id[2])
airpush_solo = visual.ResultsLoader().load_file_from_id(loop_id[3])
revmob_solo = visual.ResultsLoader().load_file_from_id(loop_id[4])


# Plot files and save in output folder
visual.Viz(all_families).plot_single('performance', fname='transcend_all.pdf') # Fig 2a 
visual.Viz(all_families).plot_single('distribution', fname='transcend_distribution_all.pdf') # Fig 2b
visual.Viz(c2, c1, label1='(C1)', label2='(C2)').plot_performance_distribution(fname='goodware_snoop_performance.pdf') # Fig 3
visual.VizExpl(cbase,c1).mean_of_weights_of_top_feature_of_missed_family_samples(['DNOTUA'], missed=True) # Table 1
visual.VizExpl(cbase,c1).mean_of_weights_of_top_feature_of_missed_family_samples(['AIRPUSH'], missed=True) # Table 3
visual.Viz(c2, c1, label1='(C2)', label2='(C1)').plot_single('difference', month_selection=[23, 31, 47], fname='goodware_snoop_diff_crop.pdf') # Figure 4a
visual.Viz(cbase, c1, label1='(Cbase)', label2='(C1)').plot_single('difference', month_selection=[25, 31, 52], fname='malware_snoop_diff_crop.pdf') # Figure 4b
visual.Viz(cbase, c1, label1='(Cbase)', label2='(C1)').plot_performance_distribution(fname='malware_snoop_performance.pdf') # Figure 5
visual.Viz(all_families_dnotua).plot_single('performance', fname='transcend_all_dnotua.pdf') # Figure 6
visual.FamilyIso(dowgin_solo,dnotua_solo,kuguo_solo,airpush_solo,revmob_solo).plot_family_iso_matrix(fname='solo_performance_grid.pdf')# Figure 7 & Table 4
visual.DimensionReduction().tsne_visual(['DOWGIN','DNOTUA','KUGUO','AIRPUSH','REVMOB'], fname='tsne_malware_overtime.pdf')
visual.VizExpl(c1).top_features_of_given_family([1,52,31,25,47,33,27,36,22,17,53,44,37],['DNOTUA']) # Table 5
visual.VizExpl(c1).top_features_of_given_family([1,52,31,25,47,33,27,36,22,17,53,44,37],['KUGUO']) # Table 6

