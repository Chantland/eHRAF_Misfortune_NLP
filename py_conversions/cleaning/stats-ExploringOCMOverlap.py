#%%
import pandas as pd                 # dataframe storing
import numpy as np
import re                           # regex for searching through strings
# import mlxtend                      # for coocurrance exploration
import copy

def OCM_stripper(df, OCM='OCM'):
    df[OCM] = df[OCM].apply(lambda x: re.sub(" |\'",'',x))
    df[OCM] = df[OCM].apply(lambda x: x[1:-1].split(','))
    return df
#%% md
# NOTE NEEDS FIXING TO UPDATE WHERE THE DATASET LIST IS TAKEN FROM
#%% md
# ## load dataframe
#%%
# CHANGE: put folder name here (use the multi-dataset code if you have multiple scrapings you wish to combine)–
folder=r'(subjects-(contracts_OR_disabilities_OR_disasters_OR_friendships_OR_gift_giving_OR_infant_feeding_OR_lineages_OR_local_officials_OR_luck_and_chance_OR_magicians_and_diviners_OR_mortuary_specialists_OR_nuclear_family_OR_priesthood_OR_prophet'   # OCM ['586' , '684' , '688' , '731' , '732' , '756' , '767' , '777' , '791' , '792' , '793' , '431' , '572' , '594' , '613' , '624' , '675' , '853']
# CHANGE: edit folder location as necessary-
directory = '../Data/'
file = '_Altogether_Dataset_CLEANED.xlsx'
file_loc = directory + '/' + folder + '/'

df = pd.read_excel(file_loc+file)
OCM_stripper(df)


#%%
df.columns
#%%
# get OCMs per dataset
datasets= df['DatasetSplitInfo'].dropna()[1:]
dataset_dict = dict()
scrapedOCMs = []
for dataset in datasets:
    dataset_name = re.findall(r'Dataset (\d+):', dataset)[0]
    OCM_list = re.findall(r'\[(.*)\]', dataset)[0]
    OCM_list = re.sub(" |\'",'',OCM_list)
    OCM_list = OCM_list.split(',')
    scrapedOCMs+=OCM_list
    dataset_dict[dataset_name] = OCM_list
# make each OCM have a paired dataset
OCMDS_dict = {value: key for key, values_list in dataset_dict.items() for value in values_list}
#%%
# get all the OCM codes and their meanings
OCM_codes = pd.read_excel('../../eHRAF_Scraper/Resources/OCM_Codes.xlsx')

df_expl = df.explode(column='OCM').reset_index(drop=True)
# Find OCM's that do not fit the normal 100-900 OCM scheme
# NOTE 0 means the material is not relevant, I am unsure, however, why this sometimes appears with other OCM's in the same passage
# NOTE I believe 5310 and 5311 are different specifications of 531 while 1710 might be a more specific (and singlular) subset of 171? I do not believe the same for 77 and 1787
list_OCM = df_expl['OCM'].value_counts().index.tolist()
OCM_freq = df_expl['OCM'].value_counts()
OCM_freq
#%%
#CHANGE overlap dataset we want to compare with. If there in none, put in None
overlapDS = '1'

columns=['OCM', 'Name', 'Freq', 'CoPrevalance_OCM', 'Buddy_OCM', 'Percentage_Overlap_with_Misf', 'Dataset']
df_OCM = pd.DataFrame(np.zeros((len(scrapedOCMs),len(columns))), columns=columns)

for index, OCM in enumerate(scrapedOCMs):

    #Get OCM Number
    df_OCM.loc[index, 'OCM'] = OCM
    
    #get OCM name
    OCM_code = OCM_codes.loc[OCM_codes['OCM']==int(OCM)]
    df_OCM.loc[index, 'Name'] = OCM_code.iloc[0]['Meaning']

    # Get OCM frequency
    df_OCM.loc[index, 'Freq'] = OCM_freq[OCM]

    # get co-prevelance (number of OCMs it appears with)
    df_subset = df.copy()
    msk = df_subset['OCM'].apply(lambda x: not set(x).isdisjoint([OCM]))
    df_subset = df_subset.loc[msk]
    df_subset['List_Length'] = df['OCM'].apply(lambda x: len(x))
    df_OCM.loc[index, 'CoPrevalance_OCM'] = round(df_subset['List_Length'].mean(),2)

    # Get OCM that is most common with the target OCM
    df_exp = df_subset.explode('OCM').reset_index(drop=True)
    OCM_Buddy = df_exp['OCM'].value_counts()
    if int(OCM_Buddy.index[0]) != int(OCM):
        raise Exception("OCM not the most frequent")
    OCM_code = OCM_codes.loc[OCM_codes['OCM']==int(OCM_Buddy.index[1])]
    df_OCM.loc[index, 'Buddy_OCM'] = OCM_code.iloc[0]['Meaning']
    # df_OCM.loc[index, 'Buddy_OCM'] = OCM_Buddy.index[1]

    # get overlap with misfortune dataset
    if overlapDS is not None:
        if OCM in dataset_dict[overlapDS]:
            df_OCM.loc[index, 'Percentage_Overlap_with_Misf'] = np.nan
        else:
            msk = df_subset['OCM'].apply(lambda x: not set(x).isdisjoint(set(dataset_dict[overlapDS])))
            df_subset = df_subset.loc[msk]
            df_OCM.loc[index, 'Percentage_Overlap_with_Misf'] = round(((len(df_subset)+.001)/OCM_freq[OCM])*100,2)

    #get Dataset of OCM
    df_OCM.loc[index, 'Dataset'] = OCMDS_dict[OCM]

df_OCM['Freq'] = df_OCM['Freq'].astype(int)
df_OCM
#%%
df_OCM.to_excel(f'{file_loc}_OCM_Overlap.xlsx')
#%%
