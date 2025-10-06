#%% md
# # Cleaning for eHRAF Scraper
# The current stats are merely to 
# - Clean and reorganize the dataframe 
# - Get datasets that correspond to OCM pairs (optional)
# - Find the most common OCM codes
# - Find association rules (when one OCM appears this other OCM is likely to appear) 
# 
#%% md
# ## Clean the Dataframe
# 
#%%
import pandas as pd                 # dataframe storing
import numpy as np
import re                           # regex for searching through strings
# import mlxtend                      # for coocurrance exploration
import copy
import os

# strip OCMs so they become a list again
def OCM_stripper(df, OCM='OCM'):
    if type(df[OCM].iloc[0]) is list: # if already a list, return without alteration
        return df
    df_ocm = df.copy() # so that the original dataframe is not affected
    df_ocm[OCM] = df_ocm[OCM].apply(lambda x: re.sub(" |\'",'',x))
    df_ocm[OCM] = df_ocm[OCM].apply(lambda x: x[1:-1].split(','))
    return df_ocm

#%% md
# ### Load data (<font color="red">Only run one cell</font>)
#%% md
# #### Load Single Scraping (Optional)
# 
#%%

# CHANGE: put folder name here (use the multi-dataset code if you have multiple scrapings you wish to combine)–
folder = r'subjects-(sickness)_FILTERS-culture_level_samples(PSF)'                          # OCM 750
# folder = r'subjects-(religious_practices_OR_sickness)_FILTERS-culture_level_samples(PSF)'   # OCM 750 + 780
# CHANGE: edit folder location as necessary-
directory = '../Data/' + folder


df_raw = pd.read_excel(directory + '/_Altogether_Dataset.xlsx')
redonePassageNums_bool = False # Here for later code which adds a second column to the dataset_lists file if needed

df_raw = OCM_stripper(df_raw)
# set up read me text for later
readme_text = 'The following post processing files (that is all folder and files excluding those produced from the scraper such as raw Culture files and _Altogether_Dataset.xlsx) were done on a single dataset/scraping\n'

print("Num Passages - ", len(df_raw))
# did it work? did it output a single OCM string?
df_raw['OCM'][0][0]

#%% md
# #### Load Multiple Scraping (Optional)
#%%
# CHANGE: put the folders of the data you wish to combine, Do not run this cell if you plan on using the cell above–

folder1 = r'subjects-(religious_practices_OR_sickness)_FILTERS-culture_level_samples(PSF)'   # OCM ['750' , '780']
folder2 = r'(subjects-(contracts_OR_disabilities_OR_disasters_OR_friendships_OR_gift_giving_OR_infant_feeding_OR_lineages_OR_etc'   # OCM ['586' , '684' , '688' , '731' , '732' , '756' , '767' , '777' , '791' , '792' , '793' , '431' , '572' , '594' , '613' , '624' , '675' , '853']
folder3 = r''
folder4 = r''   
folder_loop = [folder1, folder2, folder3, folder4]
# CHANGE: edit folder location as necessary-
directory = '../Data/'


print("Num Passages:")
readme_text = 'The following post processing files (that is all folder and files excluding those produced from the scraper such as raw Culture files and _Altogether_Dataset.xlsx) were done on a COMBINED dataset of multiple scrapings. Here are the folders used:\n'
df_raw = pd.DataFrame()
for index, folder in enumerate(folder_loop):
    if len(folder) ==0:
        continue
    file_loc = directory + folder + '/_Altogether_Dataset.xlsx'
    df_raw_input = pd.read_excel(file_loc)
    df_raw = pd.concat([df_raw, df_raw_input], ignore_index=True)
    latest_dir = directory + folder # for use of dataset later
    readme_text += f'\tfolder {index+1}: {folder}\n'
    print(f"Folder {index+1} - ", len(df_raw_input))
directory = latest_dir #make the last directory the directory that all the new files will go

# Redo Passage Numbers
redonePassageNums_bool = True # Here for later code which adds a second column to the dataset_lists file if needed
origPassageNums = df_raw['Passage Number']
df_raw['Passage Number'] = df_raw.index+1 # since there may be overlap in IDs, create new and unique passage IDs for each passage
df_raw = OCM_stripper(df_raw)


print("Total - ", len(df_raw))
#%% md
# ### Remove 'run_info' column
#%%
df_raw = df_raw.drop(['run_Info'], axis=1)
#%% md
# ### Remove blank passages
#%%
# drop all rows that have a blank passage
print(f'Before: {len(df_raw)}')
df_raw = df_raw.dropna(subset="Passage")
print(f'After: {len(df_raw)}')
#%% md
# ### Remove Duplicates
#%% md
# Currently, duplicate passages will be removed (keep only one) regardless of if they have different or same OCMs <br>
# Previously, only passages will be removed if they contain a duplicate passage with the same OCMs. Meaning duplicate passages with different OCMs would have remained
#%%
# (exploratory)  
df_dummy = copy.deepcopy(df_raw)
# Find all passages which are duplicates but do not share the same document. 
# First let's explore some of the duplicates
dup1 = df_dummy["Passage"].duplicated(keep=False)  # find all duplicate passages
dup2 = df_dummy[dup1].duplicated(subset=["Passage", "DocTitle"], keep=False) #of the duplicate passages, find those that shair a passage and doc title
# rows which contain duplicate passages but not part of the same document (only top 4 shown)
print(f'Number of passages whose duplicates come from different documents: {len(df_dummy[dup1][~dup2].sort_values(by="Passage"))}')
df_dummy[dup1][~dup2].sort_values(by='Passage').head(4) #Note that there may be 3 or more instances of a particular duplicate passage but not all may have the same doc so this line will show what seems to be a single passage that does not have a duplicate.

#%%
# (exploratory) 
# Find passages which have duplicates but whose duplicates come from different cultures
df_dummy["OCM"] = df_dummy['OCM'].apply(tuple) #turn the OCM list to a tuple to allow for comparisons

# Of the passages which have duplicates, find and keep all which have the same OCM
dup3 = df_dummy[dup1].duplicated(subset=["Passage", "OCM"], keep=False)
# Show only the passages with duplicates but NOT matching OCMs
print(f'Number of passages whose duplicates do not share OCMs:  {len(df_dummy[dup1][~dup3].sort_values(by="Passage"))}')
df_dummy["OCM"] = df_dummy['OCM'].apply(list)
df_dummy[dup1][~dup3].sort_values(by="Passage").head(4) #same quirk as above
#%%
# (exploratory, but important to check!) 
# Find passages which have duplicates but whose duplicates have different OCM numbers
df_dummy["OCM"] = df_dummy['OCM'].apply(tuple) #turn the OCM list to a tuple to allow for comparisons

# Find all passages which are duplicates but do not share the same culture. 
# First let's explore some of the duplicates
dup1 = df_dummy["Passage"].duplicated(keep=False)  # find all duplicate passages
dup2 = df_dummy[dup1].duplicated(subset=["Passage", "Culture"], keep=False) #of the duplicate passages, find those that shair a passage and doc title
# rows which contain duplicate passages but not part of the same culture
print(f'Number of passages whose duplicates come from different Cultures: {len(df_dummy[dup1][~dup2].sort_values(by="Passage"))}')
df_dummy[dup1][~dup2].sort_values(by='Passage') # NOTE Make sure these are all ones you are okay with deleting because there could be circumstances where a passage is shaired between cultures but is ultimately relevant.
#%%
# remove all duplicated passages
# df_raw["OCM"] = df_raw['OCM'].apply(tuple) #turn the OCM list to a tuple to allow for comparisons (this is here when we used to keep duplicated passages with different OCMs)

# drop duplicates
print(f'Before {len(df_raw)}')
df_raw.drop_duplicates(subset=["Passage"], keep='first', inplace=True)
print(f'After {len(df_raw)}')

# df_raw["OCM"] = df_raw['OCM'].apply(list) #turn the OCM back to a list (this is here when we used to keep duplicated passages with different OCMs)
#%% md
# ### Remove passages that are too long 
#%% md
# THIS IS RECOMMENDED FOR MACHINE LEARNING SO IF YOU DON'T CARE ABOUT THE PASSAGE BEING TOO LONG AND WANT LONG PASSAGES CODED, DO NOT RUN THIS CELL
#%%
# CHANGE the cut off at will, the max token limit for BERT is 512 but assume that some passages will have extra tokens because of weird characters or long words (etiology become et## and ##iology tokens).
cut_off = 425 

mask = df_raw['Passage'].apply(lambda x: len(x.split())<=cut_off)
print(f"Percentage of too long passages {round(100*(1 - sum(mask)/(mask.count())),1)}%")
print(f'Before {len(df_raw)}')
df_raw = df_raw.loc[mask]
print(f'After {len(df_raw)}')

#%% md
# ### Remove extra OCMs
#%%
# make the OCM into a valid format
def OCM_validity_checker(OCM_list):
    assert isinstance(OCM_list, list), "Need to insert a list"
    OCM_list = [str(OCM) for OCM in OCM_list]
    return OCM_list

# cut down and use only the OCMs we want
def OCM_remover(df, OCM_list:list, OCM_Dataset_A=False, OCM_Dataset_B=False, saveComparison=False):
    df_main = df

    # change to a list of strings if not already
    OCM_list = OCM_validity_checker(OCM_list)

    # If you use a higher order code (750) eHRAF attempts to aquire ALL OCMs related to your input.
    # select only the OCMs we originally wished to search for by inputting OCM's into a list
    msk = df_main['OCM'].apply(lambda x: not set(x).isdisjoint(OCM_list))
    df_main = df_main.loc[msk]

    print(f"Total passages after reducing\n{len(df_main)}\n")
    # If you want to compare this list with overlap (like if you have basically two searches stuck together) 
    if OCM_Dataset_A is not False and OCM_Dataset_B is not False:
        df_A = df
        df_B = df
        OCM_Dataset_A = OCM_validity_checker(OCM_Dataset_A)
        OCM_Dataset_B = OCM_validity_checker(OCM_Dataset_B)
        # get counts for both datasets
        msk = df_A['OCM'].apply(lambda x: not set(x).isdisjoint(OCM_Dataset_A))
        df_A = df_A.loc[msk]
        msk = df_B['OCM'].apply(lambda x: not set(x).isdisjoint(OCM_Dataset_B))
        df_B = df_B.loc[msk]

        msk = df_A['OCM'].apply(lambda x: not set(x).isdisjoint(OCM_Dataset_B))
        df_dummy_AB = df_A.loc[msk]

        # Go the extra step and save a comparison dataframe to excel showing the how one list's OCM's compare to the other's 
        if saveComparison is True:
            value_counts_A = df_A["Culture"].value_counts(ascending=True)
            value_counts_B = df_B["Culture"].value_counts(ascending=True)

            culture_order = value_counts_A.index

            values_A = value_counts_A.values
            values_B = value_counts_B.reindex(culture_order, fill_value=0).values #reindex so they match
            # get first value
            firstVal_A = OCM_Dataset_A[0]
            firstVal_B = OCM_Dataset_B[0]

            df_valcon =  pd.DataFrame({firstVal_A: values_A, firstVal_B: values_B}, index=culture_order)
            df_valcon["percentage"] = round(df_valcon[firstVal_A] / (df_valcon[firstVal_A] + df_valcon[firstVal_B]),2)
            df_valcon["log abs ratio"] = round(np.abs(np.log(df_valcon[firstVal_A] / df_valcon[firstVal_B])),2)
            print(f'Passages of first Dataset with the OCMs {OCM_Dataset_A}:\n{len(df_A)}')
            print(f'Passages of second Dataset with the OCMs {OCM_Dataset_B}:\n{len(df_B)}')
            print(f'First dataset\'s overlap with second dataset:\n{len(df_dummy_AB)}\n\n')
            print("Number of cultures per dataset and comparison dataframe:")
            print(df_valcon)
            df_valcon.to_excel(directory +"/_Assignment_counts.xlsx")

    else:
        print(f'Passages after reducing OCMs only to the desired number:\n{len(df_main)}')
    return df_main

#%% md
# Read the comments carefully
#%%

# CHANGE OCMs For your main filtering (MAKE SURE YOU ARE NOT RUNNING THIS UNLESS YOU AGREE WITH THE INPUTS)
# NOTE make sure you are including all OCMs you want to use in subsequent coding!!! 
# OCM_list = ["750", "751", "752", "753", "780", "781", "784", "785", '586' , '684' , '688' , '731' , '732' , '756' , '767' , '777' , '791' , '792' , '793' , '431' , '572' , '594' , '613' , '624' , '675' , '853'] 
OCM_list = ["750", "751", "752", "753", '784' , '731' , '732' , '777' , '791', '793']  #sicknes and non-sickness chosen OCMs
# OCM_list = ["750", "751", "752", "753"]  # Sickness OCMs

# (Optional) If your main dataset is comprised of multiple sub datasets, include them here merely for exploration and outputting a file which tells you the counts for each sub dataset
# ouptutes the file _Assignment_counts.xlsx
OCM_Dataset_A = ["750", "751", "752", "753"]                                                              # Sickness Dataset  
# OCM_Dataset_B = ["780", "781", "784", "785"]                                                              # 780 dataset
# OCM_Dataset_B = ['586' , '684' , '688' , '731' , '732' , '756' , '767' , '777' , '791' , '792' , '793']   # Theoretical Interest dataset
# OCM_Dataset_B = ['431' , '572' , '594' , '613' , '624' , '675' , '853']                                   # no-theoretical interest dataset
# OCM_Dataset_B = ['586' , '684' , '688' , '731' , '732' , '756' , '767' , '777' , '791' , '792' , '793' , '431' , '572' , '594' , '613' , '624' , '675' , '853'] # theoretical and non-theoretical
OCM_Dataset_B = ['784' , '731' , '732' , '777' , '791', '793']                                            # Non-Sickness dataset 


# UNCOMMENT ONE
# # Run for simple filtering (for single)
# df = OCM_remover(df_raw, OCM_list)
# Run for comparing and saving the difference between two datasets (for multi datasets)
df = OCM_remover(df_raw, OCM_list, OCM_Dataset_A, OCM_Dataset_B, saveComparison= True).copy() #copy() here to suppress the warning down below but is actually not necessary.

#%% md
# ### Shave OCMs
#%% md
# And make an exploded OCM dataframe
#%%


# Make a dataset in which each OCM have its own row by exploding (you can reset the index with .reset_index(drop=True))
df_OCM = df.explode(column='OCM').reset_index(drop=True)
# Find OCM's that do not fit the normal 100-900 OCM scheme
# NOTE 0 means the material is not relevant, I am unsure, however, why this sometimes appears with other OCM's in the same passage
# NOTE I believe 5310 and 5311 are different specifications of 531 while 1710 might be a more specific (and singlular) subset of 171? I do not believe the same for 77 and 1787
list_OCM = df_OCM['OCM'].value_counts().index.tolist()
small_OCM = [x for x in list_OCM if len(x) <3 or len(x) > 3]
print(f"OCMs too small or too large:\n{small_OCM}")

#%%
# CHANGE: add to the list for codes which should be removed (otherwise all others will be shaved)
remove_list = ['1787','7478', '77', '72']

# remove and shave OCM codes
print(f'starting list {len(df_OCM)}')
for i in remove_list:
    df_OCM = df_OCM[df_OCM["OCM"] != i]
# "Shave" the OCM codes that seem to have a parent (5310 and 5311 become 531).
df_OCM['OCM'] = df_OCM.OCM.apply(lambda x: x[0:3] if len(x) >= 3 else x)
print(f'Ending list {len(df_OCM)}')
#%%
# Apply the removals like above to the original dataframe (this is easier than just imploding as there are duplicates which limit this)
print(f'Passages before shaving and duplicate removal:\n{len(df)}')
# remove specified OCM codes
df["OCM"] = df["OCM"].apply(lambda x: [item for item in x if item not in remove_list])
# shorten the 'small_OCM' OCMs so that 5310 becomes 531
df["OCM"] = df["OCM"].apply(lambda x: [item[0:3] if item in small_OCM else item for item in x])
print(f'Passages after shaving and duplicate removal:\n{len(df)}') #Note that this number should Probably not change from the above number
# explantaion of above list comprehension: go through every row of the column "OCM" (via apply) 
# lambda x is an anonymous function which takes the row "x" and inputs it into the function.
# each row has its list items iterated over ( "___ for item in x") and checked if each list item is part of the small_OCM list, if so,
# return the first 3 characters, if not, return the original list item. Return everything back as a list and apply it to the dataframe


list_OCM = df_OCM['OCM'].value_counts().index.tolist()
small_OCM = [x for x in list_OCM if len(x) <3 or len(x) > 3]
print(f"OCMs too small or too large:\n{small_OCM}")
#%% md
# ### Create Dictionary for later count comparisons
#%%
# Find the number of passages for each culture
culture_set = set(df["Culture"])
culture_dict = {}
it_count = 0
for cult_i in culture_set:
    row_count = len(df.loc[df["Culture"]==cult_i])
    culture_dict[cult_i] = row_count
    it_count += row_count
print(f'Passages in dictionary: \n{it_count}')

#%% md
# ### Clean passage text
#%% md
# #### Remove [unknown] and [unavailable] from text
#%%
# Show the most common bracketed words (should see "[unknown]" or "[unavailable]" at the top)
def bracket_count(df=df, most_common=8):
    from collections import Counter
    bracket_set = []
    for pas in df["Passage"]:
        match = re.findall(r'\[.*?\]',pas)
        if len(match) >0:
            bracket_set += match

    counts = Counter(bracket_set)
    return counts.most_common(most_common) # N most common items
bracket_count(df,8)
#%%
# Find an example  passage to use
pattern = r'\[unavailable\]'
filtered_df = df[df['Passage'].str.contains(pattern, case=False, regex=True)]
index_pas = filtered_df.index[1]

# Example of passage which needs cleaning (May be outdated depending on your original search query!)
print(f"Before:\n {df['Passage'][index_pas]}")
# Remove all "[unknown]", "[unavailable]", "[ a ]" or "[ i ]" text within the passages"
df["Passage"] = df["Passage"].apply(lambda x: re.sub(r"\[unknown\]|\[unavailable\]|\[ a \]|\[ i \]",'', x))
# after
print(f"After:\n {df['Passage'][index_pas]}")
#%%
bracket_count(df,8)
#%% md
# ### Optional Exploration
#%%
# (OPTIONAL)
# Quick search for OCMs regardless of culture
# NOTE, sometimes a higher order code like 750 appears without lower order codes)
exclude_list = ['750','751','752','753', '784','731','732', '164','767'] #enter in the OCM strings you DON't want to see
include_list = ["624"] #enter your OCM strings of OCMs you want to see (passages only need to contain one, not all)


exc_msk = df['OCM'].apply(lambda x: set(x).isdisjoint(exclude_list))
exc = df.loc[exc_msk]
inc_msk = exc['OCM'].apply(lambda x: not set(x).isdisjoint(include_list))
inc = exc.loc[inc_msk]
inc
#%%
# Quick search for OCMs SUBINDEX BY ANOTHER COLUMN
include_list = ["159","451"] #enter your OCM strings of OCMs you want to see 
culture = "Akan" # enter the desired culture
msk = df.loc[df["Culture"]== culture]['OCM'].apply(lambda x: not set(x).isdisjoint(include_list))
out = df.loc[msk.index][msk]
out.head(4)
#%%
# (OPTIONAL)
# There are some passages that describe previous passages but do not contain information themselves like: 
# "Notes" or "End" or "Log"
# This code cell indicates (but does not remove) how many passages are short like the ones described which 
# may disrupt our OCM stats because they contain OCMs without actually having text that refers to these OCMs
shortPass_list = []
for i in df['Passage']:
    if len(i)<=10:
        shortPass_list.append(i)
print(f'Number of passages with text with 10 or fewer characters: {len(shortPass_list)}')
#%%

#%% md
# ## (OPTIONAL) Reorganize passages by source
# 
#%% md
# The following code reorganizes the passages so that sources are split up rather than clumped together in the same place.<br>
# Note that this code wile only take into account the source you want to filter. This means that the same author between different documents will be lumped sequentially together. <br>
# 
# <fontcolor=Blue>OTE:</font>
#%% md
# ### Create Inermediate dataframe (also allow for optional dataframe for testing)
#%%
# Uncomment one of these, not both

# get original dataset
df_sourceSort = df.copy()

# # OPTIONAL get dummy subsample dataset for testing
# addedPass = 0 # CHANGE this number if you want the dummy dubsample to have more rows than just 30
# x = list(range(10,20+addedPass)) + list(range(2050,2060+addedPass))+ list(range(5550,5560+addedPass))
# df_sourceSort = df.iloc[x].copy()
# df_sourceSort = df_sourceSort.sort_values(by='Passage Number')


#%% md
# ### Get optional inputs
#%%
# CHANGE enter in the column name of the type of source you want to split. Viable options are 'DocTitle', 'Section', 'Author', or really any column name you want
sourceType = 'Author'
# CHANGE enter in the number of passages per source (will loop untill all passages are organized)
n_sources = 3
# CHANGE if you are using the 'Author' source. Remove years. You can combine same named authors but from different years 
# into one source (e.g. 'Dumont, Fred, 1950-1960' and 'Dumont, Fred, 1980-2000' just become 'Dumont, Fred'
rmvAuthorYears_bool = True #change to True or False
# CHANGE Do you want the Cultures to stay in order (setting this to True organize the sources within the culture instead of the whole dataset.
# Setting this to False will interleave sources without care if they come from different cultures. Most people will probably want this to be True)
maintainCultureOrder_bool = True
#%% md
# Optionally remove years at the end of author sources 
#%%
if rmvAuthorYears_bool == True:
    if sourceType != 'Author':
        input_q = input('Your source is not the Author, are you sure you want to run this? y/n')
        if input_q.lower() != 'y':
            raise Exception('User canceled run')
        else:
            print('Author reduced')
            
    print("Unique sources BEFORE:", len(set(df_sourceSort['Author'])))
    df_sourceSort['Author']= df_sourceSort['Author'].apply(lambda x: re.sub(r',(?=[^,]*\d)[^,]*$', '', x)) # remove the end year if it has a year
    print("Unique sources AFTER:", len(set(df_sourceSort['Author'])))

#%%
df_sourceSort #show the subsample passages
#%%
# show counts for each source
df_sourceSort.value_counts(sourceType)
#%%
def sourceInterleave(df_input:pd.DataFrame, df_output:pd.DataFrame, n_sources:int):
    while len(df_input) >0:
        # Get N rows and get the index of these rows to later drop
        df_sourceSectional = df_input.groupby(sourceType).head(n_sources)
        sourceSectional_index = df_sourceSectional.index
        
        df_output = pd.concat([df_output, df_sourceSectional], ignore_index=True)

        #drop the used indexes
        df_input = df_input.drop(sourceSectional_index)
    return df_output

# n_sources = n_sources
n_sources = 3


df_output = pd.DataFrame([])
if maintainCultureOrder_bool == True: #maintain the culture order (see above 'Get optional inputs' for where this is defined)
    for cult in df_sourceSort['Culture'].unique():
        df_cult = df_sourceSort.loc[df_sourceSort['Culture']==cult].copy()
        df_output = sourceInterleave(df_input=df_cult,df_output=df_output, n_sources=n_sources)
    df_sourceSort = df_output.copy()
else:
    df_sourceSort = sourceInterleave(df_input=df_sourceSort,df_output=df_output, n_sources=n_sources)
df_sourceSort
#%%
# save new sorting to df
input_q = input('you are about to overwrite the original dataframe with a new sorting. Are you sure? y/n')
if input_q.lower() == 'y':
    df = df_sourceSort.copy()
    print('df overwritten')
else:
    print('overwrite CANCELED')
#%% md
# ## Save File
#%%
df
#%%
print(f'Passages after all cleaning:\n{len(df)}')
# Save the cleaned version of the dataframe TO THE ORIGINAL DIRECTORY
df.to_excel(directory + "/_Altogether_Dataset_CLEANED.xlsx", index=False)

# save the read me file to indicate if the cleaned file was composed of multiple datasets
# with open(directory+'/_README.txt', 'w') as f:
#     f.write(readme_text)
#%% md
# ## Differentiating Datasets
# 
#%% md
# This is so if you have multiple datasets you want to keep track of, you can print out an excel sheet which will save which passage matches with a dataset<br>
# Note that the first dataset in the heirarchy will always take precedence over lower datasets meaning that a passage which could be contained in the first and second dataset will solely be placed in the first!
#%%
def dataset_tracker(df, dataset_dict:dict):
    df_datasets = df[["Passage Number", "OCM"]]
    df_datasets.loc[:, ["Dataset","DatasetSplitInfo"]] = ''

    for key in reversed(list(dataset_dict.keys())):
        if len(dataset_dict[key]) == 0:
            continue
        
        OCM_validity_checker(dataset_dict[key])
        
        msk = df['OCM'].apply(lambda x: not set(x).isdisjoint(dataset_dict[key]))
        df_datasets.loc[msk, "Dataset"] = key

    print(f"Total: {len(df_datasets)}")
    df_datasets.loc[0, "DatasetSplitInfo"] = f"Total Count: {len(df_datasets)}"
    counter = len(df_datasets)
    for index, key in enumerate(dataset_dict.keys()):
        if len(dataset_dict[key]) == 0:
            continue
        data_count = len(df_datasets.loc[df_datasets['Dataset']==key])
        counter -= data_count
        print(f"Dataset {key}: {data_count}")
        df_datasets.loc[index+1, "DatasetSplitInfo"] = f"Dataset {key}: {dataset_dict[key]}   Count: {data_count}"

    
    if counter != 0:
        print("\n\n\033[91m{}\033[00m".format(f"WARNING number of passages do not add up to total:\n{counter}"))
        print("Make sure this is okay as it likely means your OCM codes you put in do not match the filtering done in the cleaning step")
    # If the passage numbers have been updated, add the column to the dataset as these number may be important to use later
    if redonePassageNums_bool:
        df_datasets = df_datasets.copy() #Here to suppress the slice warning but I am note sure why this fixes it.
        passLoc = df_datasets.columns.get_loc("Passage Number")
        df_datasets.insert(passLoc+1,'Passage Number Original', origPassageNums.loc[df.index])
    return df_datasets
# df['OCM'].apply(lambda x: not set(x).isdisjoint(["780", "781", "784", "785", "788"]))
    
#%%

# CHANGE insert any number of OCMs per dictionary list. You may add more datasets if you want (keep the same format shown here) or even chnage the name of the dictionary (does not have to be 1,2,3,4)
# The higher in order the list is, the more it will take precedence when deciding which dataset a passage is located in (when a passage could be contained in more than one dataset)

# # Datasets include misfortune datasets, 780 dataset, theoretical interest dataset, and non theoretical interest
# dataset_dict = {"1":["750", "751", "752", "753"], 
#                 "2":["780", "781", "784", "785"],
#                 "3":['586' , '684' , '688' , '731' , '732' , '756' , '767' , '777' , '791' , '792' , '793'],
#                 "4":['431' , '572' , '594' , '613' , '624' , '675' , '853']}

# dataset marking misfortune dataset and 
dataset_dict = {"1":["750", "751", "752", "753"], 
                "2":['784' , '731' , '732' , '777' , '791', '793'],
                "3":[],
                "4":[]}


  
df_datasets = dataset_tracker(df, dataset_dict)


# double check to make sure you are not overwriting the file you actually do not want to
print('\n')
if os.path.exists(directory+"/_Dataset_Lists.xlsx"):
    double_check = input("Are you sure you want to overwite the current dataset list? y/n")
    if double_check.lower() == 'y':
        df_datasets.to_excel(directory+"/_Dataset_Lists.xlsx", index=False)
        print('Dataset Overwritten')
    else:
        print('New dataset not saved')
else:
    df_datasets.to_excel(directory+"/_Dataset_Lists.xlsx", index=False)
    print('New dataset saved!')
#%% md
# ## (Optional) OCM Code Counting
#%% md
# Count every OCM within each culture. Do not count OCM's specified by the search (like if searched for 750-755, do not count these). 
# <!-- - REMOVE all passages which are blank since we can't very well do lexical searches on them -->
#%%
# Make a copy of df_OCM as to not interfere with other analysis
df_OCM_freq = df_OCM.copy()
# Then turn the OCM's back to an integer (for removals)
df_OCM_freq['OCM'] = df_OCM_freq.OCM.apply(lambda x: int(x))
# only keep OCMs outside our search parameters whatever those are (make sure OCM_list has been ran above)
df_sub_ex = df_OCM_freq.copy()
for OCM in OCM_list:
    df_sub_ex = df_sub_ex.loc[df_sub_ex["OCM"] != OCM]

# Overwrite and create a new dataframe for OCM counts and frequencies
df_OCM_freq = pd.DataFrame(columns=["Culture","OCM","Frequency","Proportion_of_Passages"])
for key, val in culture_dict.items():
    value_count = df_sub_ex.loc[df_sub_ex["Culture"]==key]["OCM"].value_counts()
    # duplicate the culture word and asign it to each of its rows
    cult_count = [key] * len(value_count)
    # create a culture dataframe and append it to to the 
    df_OCM_Concat = pd.DataFrame({"Culture":cult_count,"OCM":value_count.index, "Frequency":value_count.values, "Proportion_of_Passages":value_count.values/val})
    df_OCM_freq = pd.concat([df_OCM_freq, df_OCM_Concat], ignore_index=True)
df_OCM_freq = df_OCM_freq.sort_values(by = ["Culture", "Frequency"], ascending= [True, False])
df_OCM_freq
#%%
print(f'OCMs per culture: {sum(df_OCM_freq["Frequency"]) / len(set(df_OCM_freq["Culture"]))}')
#%%
# Save the file
df_OCM_freq.to_excel(directory+'/'+ "_Culture_Frequency.xlsx", index=False)
#%% md
# ## (Optional) Association Rules for OCMs
#%%
# Load resources
from mlxtend.preprocessing import TransactionEncoder

# We will use the apriori module to generate a dataframe that
# we can use for association rule finding
from mlxtend.frequent_patterns import apriori

# We will use the association_rules module to generate
# our association rules from the apriori output data frame
from mlxtend.frequent_patterns import association_rules




#%%
#Display important columns
df_smaller = df_OCM[['Culture', 'OCM','Passage']]
df_smaller
#%%
# created a grouped dataframe object by Culture and Passage 
df_group = df_smaller.groupby(by = ['Culture', 'Passage'])
df_group
#%%
def make_OCM_list(x):

    '''
    Will return a list of the unique items
    in a particular grouping when used with
    the agg method as its function
    '''

    return x.unique()
#%%
# Use the agg method and make_OCM_list
# to return a list of unique items for each ocm
# Note that depending on the filtering, there may be duplicate passages with different OCMs which are aggregated, 
# this method will combine them and extract the unique OCMs so it may not be a problem.
df_unique = df_group.agg(make_OCM_list)
#%%
list_trans = list(df_unique['OCM'])
list_trans = list_trans[0:]
len(list_trans)
#%%
te = TransactionEncoder()
encoded_itemset = te.fit(list_trans).transform(list_trans)
print(encoded_itemset.shape) # show possible transcations and number of items
te.columns_



df_encoded = pd.DataFrame(encoded_itemset, columns = te.columns_)
df_encoded.head()
#%%
# Before we begin, let's do a small
# amount of cleanup.  Let's remove all
# columns (items) that have less than 1 characters since that is just blank space
# more data cleaning my be required as time continues in case errors become evident in the scraped dataset
OCM_items = list(filter(lambda x: len(x) < 1, te.columns_ ))
print("removed: ",  OCM_items)
df_encoded = df_encoded.drop(columns=OCM_items) #remove small strings as they seem not to be items
print('How many unique items are left?', len(df_encoded.columns))
#%%
# Use apriori to create a dataframe with columns of support and itemset lists
# Note that if your items are large compared to your sample (you have few rows but many columns) I reccommend using 
# a higher min_support as many more combinations may have spuriously higher support. Also, you can crash the program if too many are selected
df_support = apriori(df_encoded, min_support=0.01, use_colnames=True)
df_support.sort_values('support', inplace=True, ascending = False)
df_support
#%% md
# ### Use association_rules to find the rules
# 
# Using the dataframe generated by `apriori`, find the association rules with the greatest lift.  See the [association_rules documentation](https://rasbt.github.io/mlxtend/api_modules/mlxtend.frequent_patterns/association_rules/) for how to do this.
# 
# Sort the resulting DataFrame by lift in descending order.  A lift > 1 indicates that the items are often purchased together and that buying X will increase the purchase of Y.  A lift of < 1 indicates the items are often substituted.  That is X is substituted for Y so X and Y don't appear together often.
# 
# Examine the resulting DataFrame.  For the association rule X -> Y, X is the column `antecedents` and Y is the column `consequents`.  If sorted you can see the metrics for each rule based upon the lift.
#%%
# Find the association rules
rules = association_rules(df_support, metric = 'lift', min_threshold=1)
# lift >1 more likely than chance X means you see Y
# lift = 1 as often as chance
# lift <1 (substitution) less likely than chance X means you see Y

#%%
# Sort the rules by lift
# and examine the output
# to find what rules were
# discovered
rules.sort_values('lift', ascending=False, inplace =True)
rules
#%%
# look for OCM codes within the list
lst = frozenset(["793","226"])
msk = rules['antecedents'].apply(lambda x: not set(x).isdisjoint(lst))
out = rules.loc[msk]
out