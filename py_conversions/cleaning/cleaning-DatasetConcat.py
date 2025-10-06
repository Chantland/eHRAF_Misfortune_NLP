#%%
import pandas as pd
import os
import numpy as np
#%% md
# # Multi Dataset Concatenator
# 
#%% md
# Combine multiple datasets into a single one for NLP predictions.
# <br><br>
#  Note that this current version is specifically made for the sickness and non-sickness combination but could be repurposed for other datasets if the code was altered to fit. Must have ran "cleaning-RACode.ipynb" with both dataset to use this notebook
#%% md
# ## Load datasets and clean to adequately concatenate
#%% md
# ### Load Datasets
#%%
# Sickness Dataset
folder = r'subjects-(sickness)_FILTERS-culture_level_samples(PSF)'  
OutputDir = '../Data/' + folder
df_1 = pd.read_excel(f"{OutputDir}/_Altogether_Dataset_RACoded.xlsx",  header=[0,1], index_col=0)

# Sickness (OCMs 750-753) and nonsickness dataset (OCMs '784, '731', '732', '777', '791', '793')
folder = r'(subjects-(contracts_OR_disabilities_OR_disasters_OR_friendships_OR_gift_giving_OR_infant_feeding_OR_lineages_OR_etc'   # OCM ['586' , '684' , '688' , '731' , '732' , '756' , '767' , '777' , '791' , '792' , '793' , '431' , '572' , '594' , '613' , '624' , '675' , '853']
OutputDir = '../Data/' + folder
df_2 = pd.read_excel(f"{OutputDir}/_Altogether_Dataset_RACoded.xlsx",  header=[0,1], index_col=0)

#dataset indicators for second dataset (used to match datasets)
df_datasets = pd.read_excel(f"{OutputDir}/_Dataset_Lists.xlsx")

#Passage Text for second dataset (used to match datasets)
df_2Uncoded = pd.read_excel(f"{OutputDir}/_Altogether_Dataset_CLEANED.xlsx")
#%%
# DELETE dummy for abby text cleaning

# rand_pass = np.random.randint(0,len(df_2)+1,30)
# df_2.iloc[rand_pass].to_excel("../../HRAF-Misf-NaturalLanguageProcessing/Abby Text Cleaning/SampleFile.xlsx")
#%%
df_2Uncoded[df_2Uncoded["Passage Number"]==1016]
#%%
df_2[df_2[("CULTURE","Passage Number")]==1016]
#%%
print("Passages in Dataset 1: ", len(df_1))
print("Passages in Dataset 2: ", len(df_2))
#Total Cultures in both datasets
print("Total Cultures in both dataset: ", len(set(df_1[("CULTURE","Culture")])|(set(df_2[("CULTURE","Culture")]))))
#%% md
# ### Match columns
#%%
# show column differences in the sickness dataset (df_1) and the combined sickness/nonsickness dataset (df_2)
set(df_1.columns).difference(set(df_2.columns))
#%%
# Alter column names to match (only done here currently as jealous_evil_eye will be converted to "other"). 
# Note that the data will be treated as the same column rather than two different columns
df_1 = df_1.rename(columns={'Jealousy_Evil_Eye':'Other'}, level=1, errors="raise")
# show again the difference (should be an empty set)
if len(set(df_1.columns).difference(set(df_2.columns))) == 0:
    print("All columns match")
else:
    print(set(df_1.columns).difference(set(df_2.columns)), "Columns unmatched")
#%% md
# ### Redo and Match Passage Numbers
# 
#%% md
# Because the sickness dataset does not contain the same passage numbers as the combined dataset (df_2), we need to redo the passage numbers to match both datasets
# 
#%% md
# #### Check For duplicates within Datasets
#%%
def duplicateCheck(df, runCol = ("CODER",'Run_Number'), dupCol = ('CULTURE','Passage'), keep='first'):
    # Check for Duplicate passages for first runthrough of this dataset (do not count extra runs, run number is updated later below but for here it should be 1)
    df = df.loc[df[runCol]==1]
    ds_duplicates = df.duplicated(subset=dupCol, keep=keep)
    if sum(ds_duplicates) >0:
        print('Duplicates:', sum(ds_duplicates) )
    else:
        print("No duplicate passages")
    return df[ds_duplicates]
duplicateCheck(df_2)
#%%
df_2[df_2[("CULTURE","Passage")].duplicated()]
#%% md
# ### Update dataframe text to match each other's exclusions
# 
#%% md
# ### Update Run Number
#%% md
# As every dataset is coded as run 1, adding multiple datasets necessitates differentiating the runs
#%%
mask = df_2[("CODER",'Run_Number')]==1
df_2.loc[mask, ("CODER",'Run_Number')] = 3 # turn all 1 runs in dataset 2 into run 3
mask = df_2[("CODER",'Run_Number')]==2
df_2.loc[mask, ("CODER",'Run_Number')] = 4 # turn all 2 runs in dataset 2 into run 4 (they probably do not exist at this moment)
#%% md
# #### Update Passage Numbers
# 
#%%
#check for incongruous passage numbers then replace them with the updated version if it is available
df_1_culture = df_1["CULTURE"].copy()
merged_df = pd.merge(df_1_culture, df_2Uncoded[['Passage Number','Passage']], on='Passage', suffixes=('_1', '_orig2'), how='left')
assert len(merged_df) == len(df_1_culture), "Error, unequal output"
incong_len = sum(merged_df["Passage Number_orig2"].isna())
print(incong_len, "passages without Passage Number")
# merged_df
#%%
# (exploratory) Check passage size for invalid ID passages, if we are dealing with the sickness dataset, the "dropped" or ingorngruent pasages should be large, say 400 words or longer
passSplit = merged_df.loc[merged_df["Passage Number_orig2"].isna()]['Passage'].apply(lambda x: len(x.split()))
print("median", passSplit.median())
print('mean', passSplit.mean())
print('min', passSplit.min())
print('max', passSplit.max())
import matplotlib.pyplot as plt
import numpy as np

# x = np.random.normal(170, 10, 250)
passSplit_maxed = passSplit.apply(lambda x: min(600, x))
plt.hist(passSplit_maxed)
plt.show()
#%%
# Give NA Passage numbers negative numbers as to not conflict with any other passage
neg_list = list(range(-1,-sum(merged_df["Passage Number_orig2"].isna())-1, -1)) #get list of negative numbers
mask = merged_df["Passage Number_orig2"].isna()
merged_df.loc[mask,"Passage Number_orig2"] = neg_list
print("NAs left:", len(merged_df[merged_df["Passage Number_orig2"].isna()]))
#%%
# update df_1 Passage number
df_1[("CULTURE","Passage Number")] =  merged_df["Passage Number_orig2"]

#%% md
# ### Concatinate both datasets
#%%
df_1and2 = pd.concat([df_1,df_2], ignore_index=True)
print('Data Length:', len(df_1and2))
#%%
# check to make sure each unique passage number has the same passage
df_1and2_dummy = df_1and2['CULTURE'].copy()
mask1 = df_1and2_dummy.duplicated(subset = "Passage Number", keep=False)
mask2 = df_1and2_dummy[mask1].duplicated(subset = "Passage", keep=False)
# df_1and2_dummy[mask1].sort_values(by="Passage Number")
assert len(df_1and2_dummy[mask1][~mask2])==0, "ERROR, Some Passage Numbers are coded for multiple unique passages, consider investigating!"
print("All duplicate Passage Numbers point to a shared Passage (this is good)")
#%% md
# ### Add Dataset Indicators
#%%
# Make sure this is the dataset which contains all other datasets
df_datasets
df_1and2dummy = df_1and2["CULTURE"].copy()
merged_df = pd.merge(df_1and2dummy, df_datasets[['Passage Number','Dataset']], on='Passage Number', suffixes=('_1and2', '_2dataset'), how='left')
assert sum(merged_df["Dataset"].isna()) == incong_len, "ERROR, Dataset contains more NA Dataset values than would be expected based on the incongruent count"
print(sum(merged_df["Dataset"].isna()), "Missing dataset indicators")
# print(sum(merged_df["Passage Number_orig2"].isna()), "passages without Passage Number")
#%%
# CHANGE New dataset indicator for passages missing one (this ideally should only be from a single dataset and corrects only the ones that were originally incongruent)
merged_df.loc[merged_df["Dataset"].isna()] = 1
print(sum(merged_df["Dataset"].isna()), "Missing dataset indicators")
# merged_df['Dataset'].value_counts(dropna=False)
#%%
assert len(merged_df) == len(df_1and2), "ERROR, Merged dataset is not the same length as the base dataset"
df_1and2[('CODER', "Dataset")] = merged_df["Dataset"]
df_1and2.head(3)
#%%
df_1and2["CODER"][["Run_Number", "Dataset"]].value_counts(sort=False, dropna=False)
#%% md
# ## Interater reliability
# 
#%%
df_1and2_dummy = df_1and2.copy()
run1_pas = set(df_1and2_dummy.loc[df_1and2_dummy[("CODER","Run_Number")]==1][("CULTURE","Passage Number")])
run2_pas = set(df_1and2_dummy.loc[df_1and2_dummy[("CODER","Run_Number")]==2][("CULTURE","Passage Number")])
run3_pas = set(df_1and2_dummy.loc[df_1and2_dummy[("CODER","Run_Number")]==3][("CULTURE","Passage Number")])
print("Number of passages contained in all 3 runs:", len(run1_pas.intersection(run2_pas).intersection(run3_pas)))
run1_pas.intersection(run2_pas).intersection(run3_pas)
#%% md
# ### Get reliability of two datasets
#%% md
# For the sake of ease (for now) we will only assume there are two runs: 1 and 3
#%%
useRuns = [1,3] #change to include the runs you want to check, it is advisable just to use two
df_CatFiltered = df_1and2.copy()
df_CatFiltered = df_CatFiltered.loc[df_CatFiltered[("CODER","Run_Number")].isin(useRuns)]
df_CatFiltered = df_CatFiltered[df_CatFiltered.duplicated(subset=("CULTURE","Passage Number"), keep=False)]


#Convert the passage numbers to the index
df_CatFiltered.loc[:, "Passage Number"] = df_CatFiltered[("CULTURE", "Passage Number")]
df_CatFiltered = df_CatFiltered.drop(columns=[("CULTURE", "Passage Number")])
df_CatFiltered = df_CatFiltered.set_index("Passage Number")

# remove extraneous columns
df_CatFiltered = df_CatFiltered.drop(columns=df_CatFiltered.columns[df_CatFiltered.columns.get_level_values(0).isin(['CULTURE','OTHER'])])
df_CatFiltered = df_CatFiltered.iloc[:, :-2] # for some reason this does not allow me to drop all of the CODER column
rem_cols = [i for i in df_CatFiltered.columns if i[1] in ['Description', 'Local_Terms', 'Local_terms', 'Other_Comments']]
df_CatFiltered = df_CatFiltered.drop(columns=rem_cols)

# check if number match above numbers
df_CatFiltered[("CODER","Run_Number")].value_counts()
#%% md
# #### Mean agreement for interater reliability
#%%
# Basic mean accuracy interater reliability
df_one = df_CatFiltered.loc[df_CatFiltered[("CODER","Run_Number")]==1][["EVENT","CAUSE","ACTION"]].sort_index()
df_three = df_CatFiltered.loc[df_CatFiltered[("CODER","Run_Number")]==3][["EVENT","CAUSE","ACTION"]].sort_index()

df_comparison = df_one == df_three

print(df_comparison.mean(axis=0))
print("\nMean Reliability:", round(df_comparison.mean(axis=0).mean(),4))
#%% md
# #### Cohen Kappa agreement by rater
#%% md
# Note that the following scores are very low, we are looking for 80% and getting around 20%. This is partly due to the inbalanced classes and the infrequentness of answering "present". It potentially could be bad that the score are so low
#%%
from sklearn.metrics import cohen_kappa_score
columns = df_one.columns
kappas_list= []
mainCol = ""
for col in columns:
    kappas = np.round(cohen_kappa_score(df_one[col], df_three[col]),4)
    if mainCol != col[0]:
        displayCol = col[0]
        mainCol = col[0]
    else:
        displayCol = len(mainCol)*' '
    kappas_list.append(kappas)
    print(f"{displayCol}{' ' *(7-len(mainCol))} {col[1]}{' ' *(25-len(col[1]))} {kappas}")
print("\nMean Kappas:", round(np.mean(kappas_list),4))
#%%
# DUMMY DELETE use sample data for cohen
# off by 1 in a inbalanced dataset, mean accuracy
list_1 = np.array([1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0])
list_2 = np.array([0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0])
cohen = np.round(np.array(cohen_kappa_score(list_1, list_2)),4)
print("Accuracy:", (np.mean(list_1==list_2)), "Cohen:", cohen )

# off by 1 in a balanced dataset (8 zeroes and 8 ones)
list_1 = np.array([1,1,0,1,1,0,0,1,1,1,0,0,1,0,0,0])
list_2 = np.array([0,1,0,1,1,0,0,1,1,1,0,0,1,0,0,0])
cohen = np.round(np.array(cohen_kappa_score(list_1, list_2)),4)
print("Accuracy:", (np.mean(list_1==list_2)), "Cohen:", cohen )
#%%
# Calculate Kappas raw


from sklearn.metrics import confusion_matrix

# column = ("EVENT","Accident")
column = ("EVENT","No_Info")

conf = confusion_matrix(df_one[column], df_three[column])
print(conf)
tn, fp, fn, tp = conf.ravel()

p0 = (tn+tp)/np.sum(conf)
print("P0:", p0)
p_cor = ((tn+fp)/np.sum(conf)) * ((tn+fn)/np.sum(conf))
p_inc = ((fn+tp)/np.sum(conf)) * ((fp+tp)/np.sum(conf))
pe = p_cor+p_inc
print("Pe:",pe)
k =  (p0 - pe)/(1-pe)
print("K:", k)

#%%
p0 = (tn+tp)/np.sum(conf)
print("P0:", p0)
p_cor = ((tn+fp)/np.sum(conf)) * ((tn+fn)/np.sum(conf))
p_inc = ((fn+tp)/np.sum(conf)) * ((fp+tp)/np.sum(conf))
pe = p_cor+p_inc
print("Pe:",pe)
k =  (p0 - pe)/(1-pe)
print("K:", k)
#%% md
# ## Save dataset
#%%
# CHANGE Dataset info as appropriate!
assert len(df_1and2) == 11005, "WARNING, it looks like you have made changes or additions to the dataset since last this was ran, please make sure the below info is correct and then edit this line with the new correct dataset length!"
info = df_datasets.loc[df_datasets["DatasetSplitInfo"].str.contains('Dataset') == True]["DatasetSplitInfo"].copy()
df_1and2[("CODER","Info")] = np.nan
df_1and2[("CODER","Info")] = df_1and2[("CODER","Info")].astype('object') # here to avoid deprecation of float64
df_1and2.loc[0:1, ("CODER","Info")] = list(info)
df_1and2.loc[sum(~df_1and2[("CODER","Info")].isna()), ("CODER","Info")] = "Run 1: Spring 2023 Coding of Sickness dataset (Dataset 1)"
df_1and2.loc[sum(~df_1and2[("CODER","Info")].isna()), ("CODER","Info")] = "Run 2: Summer 2023 Recoding of Sickness dataset (Dataset 1)"
df_1and2.loc[sum(~df_1and2[("CODER","Info")].isna()), ("CODER","Info")] = "Run 3: Fall 2023-Spring 2024 recoding of Sickness dataset (Dataset 1) and non-Sickness dataset (Dataset 2)"
#%%

df_1and2.to_excel(f"{OutputDir}/_Altogether_Dataset_RACoded_Combined.xlsx")