#%% md
# # Cultural Splitter
#%% md
# For taking cleaned scraped data file ("_Altogether_Dataset_CLEANED.xlsx" which is made using stats-eHRAF_Scraper.ipynb) that contains ALL cultures and splitting this file into individual cultural datafiles (with formatting) for each culture. The purpose is to have guidelines and colors to help the Research Assistants code the passages<p>
# <br> Additionally, optionally only take a smaller subset of the passsages based on a desired distribution of predicted labels (that is, if we believe OCM 751 related to our desired label EVENT_Illness, we might try to get a subset of the passages which contain the OCM 751 50% of the time)
# <br><br>
# <font color="red"> Be careful with running cells where it is noted in red</font>
#%%
import pandas as pd
import numpy as np
import xlsxwriter
import math
import os
import re
def OCM_stripper(df, OCM='OCM'):
    if type(df[OCM].iloc[0]) is list: # if already a list, return without alteration
        return df
    df_ocm = df.copy() # so that the original dataframe is not affected
    df_ocm[OCM] = df_ocm[OCM].apply(lambda x: re.sub(" |\'",'',x))
    df_ocm[OCM] = df_ocm[OCM].apply(lambda x: x[1:-1].split(','))
    return df_ocm
#%%
# CHNAGE: put folder name here–
# folder = r'subjects-(religious_practices_OR_sickness)_FILTERS-culture_level_samples(PSF)' # original 750
folder = r'(subjects-(contracts_OR_disabilities_OR_disasters_OR_friendships_OR_gift_giving_OR_infant_feeding_OR_lineages_OR_etc' # non-misfortune scraping but contains 


# CHANGE: edit folder location as necessary-
directory = '../Data/' + folder
# CHANGE: culture_folder if you want a different name for the cultures to be deposited in
culture_folder = "Cleaned_Cultures"


df = pd.read_excel(directory + "/_Altogether_Dataset_CLEANED.xlsx")
df = OCM_stripper(df)

# drop columns list, Columns which no longer exist or are deprecated will be skipped.
drop_cols = ['run_Info']

# optionally load in the dataset indicator if it is there then append it to the dataframe
useDatasets = False
try:
    df_dataset = pd.read_excel(directory + "/_Dataset_Lists.xlsx")
    assert len(df_dataset) == len(df)
except AssertionError:
    print("""\033[91mWARNING\033[00m Dataset indicator length different from main Dataset, 
this may mean your _Dataset_Lists file was created from a different subset 
as your _CLEANED_Altogether_Dataset.xlsx. If this was intentially, it may be 
good to delete _Dataset_Lists.xlsx as this can cause confusion if left in.""")
except:
    print('No dataset indicators loaded')
else:
    # get Dataset_lists
    datasets= df_dataset['DatasetSplitInfo'].dropna()[1:]
    dataset_dict = dict()
    scrapedOCMs = []
    for dataset in datasets:
        dataset_name = re.findall(r'Dataset (\d+):', dataset)[0]
        OCM_list = re.findall(r'\[(.*)\]', dataset)[0]
        OCM_list = re.sub(" |\'",'',OCM_list)
        OCM_list = OCM_list.split(',')
        scrapedOCMs+=OCM_list
        dataset_dict[dataset_name] = OCM_list
    df_dataset = df_dataset[['Passage Number', 'Dataset']]
    df_dataset['Dataset'] = df_dataset['Dataset'].astype(str) # for easier indexing in the dictionary created above
    df = df.merge(df_dataset, on='Passage Number', how='left') # add the dataset column to the main dataframe (merging in case some cultures got moved around)

    drop_cols+= ['Dataset'] # delete the dataset column later
    useDatasets = True
    print("Dataset indicators successfully loaded")


#%%
def make_dir(path, exist_ok=False): #make directory
    import os
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if exist_ok==True or not isExist:
    # Create a new directory because it does not exist
        os.makedirs(path, exist_ok=exist_ok)
    else:
        print('Warning, folder already exists and therefore new folder was not created. Set exist_ok=True if you want to disregard this')
        

path = directory + '/' + culture_folder
# Make folder
make_dir(path)
#%% md
# ## (optional) Stratified random sampling for balanced datasets
#%% md
# The following code is optional. Only run if your loaded dataset is actually comprised of 2 datasets AND you wish to balance these two by OCMs. It constructs the corpus to be composed of balanced distributions of OCMs from target dataset(s). This means that you should have somewhat more equal number of both datasets. This will also maintain the balance of the OCMs in the target dataset but WON'T maintain the balance of the reference dataset (usually misfortune)
#%% md
# <font color='red'>NOTE, if you are using the following scraping: <br><font size= 1px>r'(subjects-(contracts_OR_disabilities_OR_disasters_OR_friendships_OR_gift_giving_OR_infant_feeding_OR_lineages_OR_local_officials_OR_luck_and_chance_OR_magicians_and_diviners_OR_mortuary_specialists_OR_nuclear_family_OR_priesthood_OR_prophet' [[[[or]]]] r'(subjects-(contracts_OR_disabilities_OR_disasters_OR_friendships_OR_gift_giving_OR_infant_feeding_OR_lineages_OR_etc'</font><br> Due to an error with not setting the seed of random sampling, this dataframe used an unknown different seed than the one in this dataframe, therefore to prevent overlap, it is advisable  to skip all of this and go to the Emergency Retrieval of Cultures section if you wish to replicate the files. Note that backup files in case of an overwrite can be found in the folder "_SaveFromOverwrite". For every other dataset, use this code normally.<br>See [Emergency Retrieval of Cultures](#emergency-retrieval-of-cultures) below</font>
#%%
assert useDatasets==True, "You do not have dataset indicators loaded"

# CHANGE: enter in the key for the dictionary that contains your datasets (usually is '1' and '2')
ds_ref = '1' # this dataset will be used as the reference that the other dataset will try to match in its count
ds_tar = '2' 

# Create a column for if we should include the passage or not, then include all passages in the reference dataset
df['Include'] = 0
msk = df['OCM'].apply(lambda x: not set(x).isdisjoint(dataset_dict[ds_ref]))
df.loc[msk, 'Include'] = 1

assert len(df[msk]) == sum(df['Include']), "Include dataset does not match the totality of the reference dataset"

#%%

df_OCMExplode = df.explode(column='OCM').reset_index(drop=True)
#get value count for all OCMs regardless of its inclusion
valcount_OCMCultures = df_OCMExplode.value_counts(subset=['Culture','OCM'])
# get value counts with the reference dataset removed (used later for selecting subset)
valcount_OCMCultures_target = df_OCMExplode[df_OCMExplode['Include']==0].value_counts(subset=['Culture','OCM'])
valcount_OCMCultures
#%%
cultures = valcount_OCMCultures.index.get_level_values('Culture').unique()
df_Freqs = pd.DataFrame(0,index=cultures, columns=["ReferenceDS_Freq","TargetDS_Freq_Raw", "TargetDS_Freq_Redux"]+scrapedOCMs)
# df_Freqs = pd.DataFrame(np.nan, index=cultures, columns=["ReferenceDS_Freq","TargetDS_Freq"]+scrapedOCMs)
df_Freqs.head(4)
#%%
# Get frquency counts
 # Get a column ready for marking which passages will be included (the target dataset will be assumedly reduced in favor of balancing it with the reference dataset)
df_include = df.copy()
for culture in cultures:
    valCnt_Cult = valcount_OCMCultures[culture] # make it less wordy
    valCnt_Cult_tar = valcount_OCMCultures_target[culture]
    # get reference frequency
    presentRefOCM = list(set(dataset_dict[ds_ref]) & set(valCnt_Cult.index))
    df_cult = df.loc[df['Culture']==culture]
    msk = df_cult['OCM'].apply(lambda x: not set(x).isdisjoint(presentRefOCM))
    df_cult = df_cult.loc[msk]
    ReferenceDS_Freq = len(df_cult)
    df_Freqs.loc[culture,'ReferenceDS_Freq'] = ReferenceDS_Freq

    # get target frequency before reduction
    targetRefOCM = list(set(dataset_dict[ds_tar]) & set(valCnt_Cult_tar.index)) # get all OCMs that exist within the dataset and within the frequency count for this culture
    targetRefOCM_freq = sum(valCnt_Cult_tar[targetRefOCM]) # get the total OCM counts (used later)
    df_cult = df.loc[(df['Culture']==culture) & (df['Include']==0)]
    tar_msk = df_cult['OCM'].apply(lambda x: not set(x).isdisjoint(targetRefOCM))
    df_cult = df_cult.loc[tar_msk]
    TargetDS_Freq_Raw = len(df_cult)
    df_Freqs.loc[culture,'TargetDS_Freq_Raw'] = TargetDS_Freq_Raw 


    #get individual OCM frequencies for reference (these will not be reduced)
    OCM_freq = [valCnt_Cult[OCM] for OCM in presentRefOCM]
    df_Freqs.loc[culture,presentRefOCM] = OCM_freq


    # get individual OCM frequencies for target (these may be reduced)
    OCM_freq = np.array([valCnt_Cult_tar[OCM] for OCM in targetRefOCM])
    # Reduce by a comenserate amount if the target passages are greater than the reference passages. Otherwise keep the counts as they are
    if ReferenceDS_Freq < TargetDS_Freq_Raw:
        OCM_freq = ReferenceDS_Freq * (OCM_freq/targetRefOCM_freq)
        reducedOCM_freq = np.ceil(OCM_freq)
        # Ceiling will almost always make the target dataset larger than the reference. 
        # This is problematic when the reference dataset is too small to have a distribution of target OCMS 
        # (e.g., when the reference has only 1 passage, what target OCM do you use to match with it?)
        # Therefore, reduce the count by one for the OCMs closest to rounding down.
        if sum(reducedOCM_freq) > ReferenceDS_Freq:
            difference = int(sum(reducedOCM_freq) - ReferenceDS_Freq)
            OCM_freq_mod = OCM_freq % 1
            sorted_indices = np.argsort(OCM_freq_mod)
            filtered_indices = [i for i in sorted_indices if OCM_freq_mod[i] > 0]
            lowest_X_indices = filtered_indices[:difference] # Select the first X filtered indices based on the difference
            reducedOCM_freq[lowest_X_indices] -= 1
        OCM_freq = reducedOCM_freq
    # assign the rows randomly for inclusion based on the fractions above
    cult_msk = df_include['Culture'] == culture
    for freq, OCM in zip(OCM_freq, targetRefOCM):
        ocm_msk = df_include['OCM'].apply(lambda x: not set(x).isdisjoint([OCM]))
        inc_msk = df_include['Include'] == 0
        freq -= len(df_include.loc[cult_msk & ocm_msk & (df_include['Include']==1) & ~(df_include['Dataset']==ds_ref)]) # remove the frequency of OCMs currently already in the included dataset but do not include those found in the reference dataset
        if len(df_include.loc[ocm_msk & inc_msk & cult_msk]) < freq:
            sample_num = len(df_include.loc[ocm_msk & inc_msk & cult_msk])
        else:
            sample_num = int(freq)
        # If the frequency is positive, add it
        if sample_num >0:
            random_passages = df_include.loc[ocm_msk & inc_msk & cult_msk].sample(sample_num, random_state=10)
            df_include.loc[df['Passage Number'].isin(random_passages['Passage Number']), 'Include'] = 1
        # indices_to_mark = np.random.choice(df_dummy.loc[ocm_msk & inc_msk & cult_msk].index, sample_num, replace=False)
        # df_dummy.loc[indices_to_mark, 'Include'] = 1
    df_Freqs.loc[culture,targetRefOCM] = OCM_freq
print("Dataframe of OCM frequency before reduction (note OCM frequencies only taken from reference)")
df_Freqs.head(5)
# df_include['Culture'].value_counts()

#%% md
# Update the cultural frequency with the reduced list of OCMs. Inspect the two dataframes and make sure that the OCM frequencies between the above dataframe are similar to the below (it is probably not a perfect match) <br>
# Note that the OCMS may seem rather large as they are shared between two datasets, the reference dataset and the target. The target Dataset is all passages NOT contained within the reference dataset. This means that the OCMs for the target dataset may bleed into the reference but not vice vera. For the sake of the current experiment plan, this means that one dataset is the sickness dataset which may contain some OCMs from the non-sickness dataset
#%%
# Do the cultural loop all over again but this time refresh the count for OCMS for redux
df_Freqs_updated = df_Freqs.copy()
# get target frequency AFTER reduction
df_OCMExplode = df_include.explode(column='OCM').reset_index(drop=True)
valcount_OCMCultures_target_redux = df_OCMExplode[df_OCMExplode['Include']==1].value_counts(subset=['Culture','OCM'])
for culture in cultures:
    valCnt_Cult_tarRedux = valcount_OCMCultures_target_redux[culture]
    targetRefOCM = list(set(dataset_dict[ds_tar]) & set(valCnt_Cult_tarRedux.index)) # get all OCMs that exist within the dataset and within the frequency count for this culture
    targetRefOCM_freq = sum(valCnt_Cult_tarRedux[targetRefOCM])  # get sum counts of all target OCMs within the frequency count of this culture
    OCM_freqRedux = np.array([valCnt_Cult_tarRedux[OCM] for OCM in targetRefOCM])
    df_Freqs_updated.loc[culture,targetRefOCM] = OCM_freqRedux

    cult_msk = df_include['Culture']==culture
    inc_msk = df_include['Include']==1
    ocm_msk = df_include['OCM'].apply(lambda x: not set(x).isdisjoint(targetRefOCM))
    df_Freqs_updated.loc[culture,'TargetDS_Freq_Redux'] = len(df_include.loc[ocm_msk & inc_msk & cult_msk & (df_include['Dataset']==ds_tar)])
df_Freqs_updated.head(5)

#%%
# check if there really is no OCMs from the reference in the target dataset (this should show no passages below)
msk = df_cult['OCM'].apply(lambda x: not set(x).isdisjoint(dataset_dict[ds_ref]))
df_include.loc[(df_include['Dataset']=='2')&(df_include['Include']==1)&msk]
#%% md
# Save frequency dataset and save a list of inclusions to the Dataset file
#%%
# set up df_dataset (again)
assert useDatasets==True, "You do not have dataset indicators loaded"
df_dataset = pd.read_excel(directory + "/_Dataset_Lists.xlsx")
assert len(df_dataset) == len(df), "lengths do not match up. Check for discrepancies"
assert sum(df_dataset['Passage Number'] - df['Passage Number'])==0, "Passage Numbers do not match up"

# append or update inclusion
if 'Include' in df_dataset.columns:
    df_dataset["Include"] = df['Include']
else:
    dataset_colNum = df_dataset.columns.get_loc('Dataset')
    df_dataset.insert(dataset_colNum+1,'Include', df['Include'])


df_Freqs_updated.to_excel(directory+'/Cleaned_Cultures/'+'_Culture_Split_Count.xlsx', index=False)
df_dataset.to_excel(directory + "/_Dataset_Lists.xlsx", index=False)
#%% md
# ## Partition DataFrame for Large Passage Counts
# 
#%% md
# As some cultures have so many passages it may be unlikely for one RA to finish them, it can be important to partition the cultures. The downside is that the balance of the dataframe in terms of OCMs will be off but that would have already been the case should a culture be incomplete and require multiple coders
#%%
def partioner(df, cultureCol, partition_size=300,  threshold_size=100):
    df_partition = pd.DataFrame(columns=['Culture','Partition_Num', 'Total_size', 'Partition_Size', 'Partition_Bool', 'Start', 'End'])
    culture_set = set(df[cultureCol])
    for culture in culture_set:
        df_cult = df.loc[df[cultureCol]==culture]
        if len(df_cult) < partition_size+threshold_size: # if it is already too small, do not subdivide
            df_partition.loc[len(df_partition.index)] = [culture, 1, len(df_cult), len(df_cult), False, 0, len(df_cult)]
        else:
            #get partition count, if the remainder is above the threshhold, increase the part by 1
            part_num, remainder = divmod(len(df_cult), partition_size)
            if remainder >= threshold_size:
                part_num = part_num+1
            # append partitions to list
            part_list = np.array_split(np.arange(0, len(df_cult)), part_num)
            for i, part in enumerate(part_list):
                df_partition.loc[len(df_partition.index)] = [culture, i+1, len(df_cult), len(part), True, part[0], part[-1]]
    return df_partition
#%%
# CHANGE THESE AS NEEDED
partition_size = 300 # What is the ideal passage number for partitions
threshold_size = 100 # what is the number where once a modulus is above this number, an additional partition is done 
                    # (like if your partions are 300 and the threshold was 100 then a culture with 690 passages would 
                    # be split 2 times while 700 would be split  three times)
df_partition = partioner(df, cultureCol="Culture", partition_size=partition_size,  threshold_size=threshold_size)
df_partition.to_excel(directory+'/Cleaned_Cultures/'+'_Partitions.xlsx')
#%% md
# ## Split Cultures
#%% md
# ### Set up columns for later splitting
#%%
# beta for automatically adding headers and coloring them
# Note that this nonIndex is old and not the current coding frame used
df_col = pd.read_excel("CodingFrame_nonindex.xlsx")
df_col = [re.sub("EVENT_|CAUSE_|ACTION_", '', i) for i in df_col]
# df_col = [re.sub("(No_Info).*", '', i) for i in df_col]
df_append= pd.DataFrame(columns=df_col)
del df_col

df_append.head(4)

# df_col

#%%
# delete uneeded extra columns as this is not necessary
for cols in drop_cols:
    try:
        df = df.drop(columns=cols)
    except:
        print(f"{cols} could not be dropped (potentially does not exist)")
# df = df.drop(columns=drop_cols)
#%%
# Assign and create multiindex length
Culture_col = len(df.columns)
Event_num = 6
Cause_num = 9
Action_num = 9
Other_num = 1
# create multiindex
multindex = ["CULTURE"]*Culture_col + ["EVENT"] * Event_num + ["CAUSE"] * Cause_num + ["ACTION"] * Action_num + ["OTHER"] * Other_num


# concat dataframes
df_coding = pd.concat([df, df_append], axis=1)
# assign levels and create multiindex
levels = [multindex, df_coding.columns]
columns = pd.MultiIndex.from_arrays(levels)
df_coding.columns = columns
df_coding.head(4)
#%%
# Blank out the index for easier understanding when exporting
x = len(df_coding) * [' ']
df_coding[" "] = x
df_coding = df_coding.set_index(" ")
#%% md
# ### Create cleaned and refitted culture cells for later coding
#%%
# specify colors
brown = "#BF6F24"
purple = "#AA3AB3"
blue = "#2986A0"
white = "#FFFFFF"
grey = "#D9D9D9"
darkgrey = "#747474"



# Reformat culture file
for i in df_partition.index:
    # Create updated name based on if there is a partition
    culture = df_partition['Culture'][i]
    culture_fileName = culture
    if df_partition['Partition_Bool'][i]:
        part_name = f"_Part{df_partition['Partition_Num'][i]}"
        culture_fileName = culture_fileName+part_name

    writer = pd.ExcelWriter(path + "/" + culture_fileName + ".xlsx", engine='xlsxwriter')
    df_cult = df_coding.loc[df_coding[("CULTURE","Culture")]==culture]
    start = df_partition['Start'][i]
    end = df_partition['End'][i]+1
    df_cult.iloc[start:end].to_excel(writer, sheet_name='Sheet1', index=True, header=True)
    workbook  = writer.book
    worksheet = writer.sheets['Sheet1']


    # align and expand the coding columns
    center_cells = workbook.add_format({'align': 'center', 'valign':'vcenter'})
    col_num = 13
    worksheet.set_column(col_num,len(df_coding.columns.get_level_values(1))+1,17, center_cells)

    # enlarge passage column
    word_wrap = workbook.add_format()
    word_wrap.set_text_wrap()
    col_idx = df_coding["CULTURE"].columns.get_loc("Passage") + 1
    worksheet.set_column(col_idx,col_idx,90, word_wrap)

    # enlarge shaman cell to be a little more accommodating
    col_idx = df_coding.columns.get_level_values(1).get_loc("Shaman_Medium_Healer") +1
    worksheet.set_column(col_idx,col_idx,20)


    # set format for each header
    culture_header = workbook.add_format({'bold': True, 'bg_color': grey, 'align': 'center', 'border':1})
    Event_header = workbook.add_format({'bold': True, 'bg_color': blue, 'align': 'center', 'font_color':white, 'border':1})
    Cause_header = workbook.add_format({'bold': True, 'bg_color': brown, 'align': 'center', 'font_color':white, 'border':1})
    Action_header = workbook.add_format({'bold': True, 'bg_color': purple, 'align': 'center', 'font_color':white, 'border':1})
    Other_header = workbook.add_format({'bold': True, 'bg_color': darkgrey, 'align': 'center', 'font_color':white, 'border':1})
    # make a list which we will index for the headers
    header_list = [culture_header]*Culture_col + [Event_header] * Event_num + [Cause_header] * Cause_num + [Action_header] * Action_num + [Other_header] * Other_num
    for col_num, value in enumerate(df_coding.columns.values):
        worksheet.write(0, col_num + 1, value[0], header_list[col_num])
        worksheet.write(1, col_num + 1, value[1], header_list[col_num])
        # worksheet.set_column()
    # freeze the top 2 rows
    worksheet.freeze_panes(2, 0)
    writer.close()


#%%

# culture_set= set(df["Culture"])
# # create a culture file
# for culture in culture_set:
#     df.loc[df["Culture"]==culture].to_excel("Cleaned_Cultures/" + culture + ".xlsx", index=False, header=True)
#%%
# Check
df.loc[df["Culture"]==culture].iloc[0]["OWC"]
#%% md
# ## Cultural Counts Dataframe
#%%
# create culture counts dataframe
df_culture = pd.DataFrame({"OWC":[], "Region":[], "Culture":[], "Partition":[], "Total Passage Count":[], "Partitioned Passage Count":[]})
df_culture[["Culture", 'Partition', "Total Passage Count", "Partitioned Passage Count"]] = pd.concat([df_partition['Culture'],  df_partition['Partition_Num'],  df_partition['Total_size'], df_partition['Partition_Size']],axis=1)

df_reduced = df[["OWC", "Culture", "Region"]].drop_duplicates()
for i, rows in df_partition.iterrows():
    culture = rows["Culture"]
    OWC = df_reduced.loc[df_reduced["Culture"]==culture].iloc[0]["OWC"]
    Region = df_reduced.loc[df_reduced["Culture"]==culture].iloc[0]["Region"]
    df_culture.loc[i, ["OWC","Region"]] = [OWC, Region]
df_culture
#%%
df_culture = df_culture.sort_values(by="Total Passage Count")
df_culture.to_excel(path+"/_CultureCounts.xlsx", index=False)

#%%
df["Culture"].value_counts(ascending=True)
#%% md
# ## Emergency Retrieval of Cultures
#%% md
# <font color='red'>Only to fix the random seed mistake of the combined dataset. DO NOT RUN THESE CELLS unless you are sure you know how it works. This only works for the Fall 2023 Semester (and potentially Spring 2024)</font>
#%%
# # Sickness (OCMs 750-753) and nonsickness dataset (OCMs '784, '731', '732', '777', '791', '793')
RA_list = ["AM","AH","CHD","JM","KB","KY","LR","MJ","SC", "_Cultures_Uncoded"] # RAs Fall 2023
RACodingDir = '../RA Coding/Coding/'
folder = r'(subjects-(contracts_OR_disabilities_OR_disasters_OR_friendships_OR_gift_giving_OR_infant_feeding_OR_lineages_OR_etc'   # OCM ['586' , '684' , '688' , '731' , '732' , '756' , '767' , '777' , '791' , '792' , '793' , '431' , '572' , '594' , '613' , '624' , '675' , '853']
OutputDir = '../Data/' + folder


def cultFileExtract(directory, finished:bool=False) -> pd.DataFrame:
    # Extract file 
    df_coding = pd.DataFrame()
    if os.path.exists(directory) == False:
        return None

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if all([os.path.isfile(f), filename.endswith(".xlsx")]) and not any([filename.startswith("_"), filename.__contains__("Eric"), filename.__contains__("recode"), filename.__contains__("Errors"), filename.startswith("~$")]):
            df_coding_culture = pd.read_excel(f, header=[0,1])
            # Mark if 1st of 2nd run
            if f.__contains__("_2ndRun"):
                df_coding_culture[("CODER","Run_Number")] = 2
            else:
                df_coding_culture[("CODER","Run_Number")] = 1
            df_coding = pd.concat([df_coding, df_coding_culture], ignore_index=True)
            print(f"Used: {f}")
    if len(df_coding) > 0: #if any file was loaded, mark True or false for finished
        df_coding[("CODER","Finished")] = finished
    return df_coding.copy()

df_coding = pd.DataFrame()
for RA in RA_list:
    # Extract unfinished 
    directory = RACodingDir + RA
    df_coding_RA_unfin = cultFileExtract(directory, finished=False)
    # Extract Finished
    directory += '/Finished'
    df_coding_RA_fin = cultFileExtract(directory, finished=True)

    # concatinate the finished and unfinished then set the column by name
    df_coding_RA_fin = pd.concat([df_coding_RA_fin, df_coding_RA_unfin], ignore_index=True)
    df_coding_RA_fin[("CODER","Coder")] = RA
    df_coding = pd.concat([df_coding, df_coding_RA_fin], ignore_index=True)


# Drop empty first column and empty first row(s)
df_coding = df_coding.drop(columns=df_coding.columns[0])
df_coding = df_coding.dropna(subset=[("CULTURE","Passage Number")])
#OCM strip
df_coding = OCM_stripper(df_coding, OCM=('CULTURE','OCM')); 
df_coding= df_coding.astype({('CULTURE','Passage Number'): 'int32'})
print(f"Unfinished Cultures: {len(set(df_coding[~df_coding[('CODER','Finished')]][('CULTURE','Culture')]))}")
print(f"Finished Cultures: {len(set(df_coding[df_coding[('CODER','Finished')]][('CULTURE','Culture')]))}")
print(f"DATAFRAME ROWS: {len(df_coding)}")
print('Note that partitions may inflate the number of \"finished\" culture counts')
#%%
# CHNAGE: put folder name here–
# folder = r'subjects-(religious_practices_OR_sickness)_FILTERS-culture_level_samples(PSF)' # original 750
folder = r'(subjects-(contracts_OR_disabilities_OR_disasters_OR_friendships_OR_gift_giving_OR_infant_feeding_OR_lineages_OR_etc' # non-misfortune scraping but contains 
# CHANGE: edit folder location as necessary-
directory = '../Data/' + folder
# CHANGE: culture_folder if you want a different name for the cultures to be deposited in
culture_folder = "Cleaned_Cultures"


df = pd.read_excel(directory + "/_Altogether_Dataset_CLEANED.xlsx")
df = OCM_stripper(df)

# drop columns list
drop_cols = ['run_Info']

# optionally load in the dataset indicator if it is there then append it to the dataframe
useDatasets = False
try:
    df_dataset = pd.read_excel(directory + "/_Dataset_Lists.xlsx")
    assert len(df_dataset) == len(df)
except AssertionError:
    print("""\033[91mWARNING\033[00m Dataset indicator length different from main Dataset, 
this may mean your _Dataset_Lists file was created from a different subset 
as your _CLEANED_Altogether_Dataset.xlsx. If this was intentially, it may be 
good to delete _Dataset_Lists.xlsx as this can cause confusion if left in.""")
except:
    print('No dataset indicators loaded')
else:
    # get Dataset_lists
    datasets= df_dataset['DatasetSplitInfo'].dropna()[1:]
    dataset_dict = dict()
    scrapedOCMs = []
    for dataset in datasets:
        dataset_name = re.findall(r'Dataset (\d+):', dataset)[0]
        OCM_list = re.findall(r'\[(.*)\]', dataset)[0]
        OCM_list = re.sub(" |\'",'',OCM_list)
        OCM_list = OCM_list.split(',')
        scrapedOCMs+=OCM_list
        dataset_dict[dataset_name] = OCM_list
    df_dataset = df_dataset[['Passage Number', 'Dataset']]
    df_dataset['Dataset'] = df_dataset['Dataset'].astype(str) # for easier indexing in the dictionary created above
    df = df.merge(df_dataset, on='Passage Number', how='left') # add the dataset column to the main dataframe (merging in case some cultures got moved around)

    drop_cols+= ['Dataset'] # delete the dataset column later
    useDatasets = True
    print("Dataset indicators successfully loaded")
#%% md
# #### Clean the ends of dataframe
# 
# 
# 
# 
#%%
# May need to erase columns as fit
dropcolumns = ["CODER"]
df_coding = df_coding.drop(columns=dropcolumns, axis=1, level=0) #The "axis" and "level" Parameters are not necessary in this case but help speeds (and get rid of warnings) when using more complicated dropping like dropping by multiindex not multiheader)
df_coding.head(2)

#%% md
# Get partitions [run the partitioner function above but do not run anything else](#partition-dataframe-for-large-passage-counts)
#%%
# Get partitions

# CHANGE THESE AS NEEDED
partition_size = 300 # What is the ideal passage number for partitions
threshold_size = 100 # what is the number where once a modulus is above this number, an additional partition is done 
                    # (like if your partions are 300 and the threshold was 100 then a culture with 690 passages would 
                    # be split 2 times while 700 would be split  three times)
df_partition = partioner(df_coding, cultureCol=("CULTURE","Culture"), partition_size=partition_size,  threshold_size=threshold_size)
assert len(df_coding)==11601, "dataframe length does not match original dataframe for this dataset. Please be careful you are not running this code when you shouldn't"
df_partition.to_excel(directory+'/Cleaned_Cultures/'+'_Partitions.xlsx')
#%%
# Blank out the index for easier understanding when exporting
x = len(df_coding) * [' ']
df_coding[" "] = x
df_coding = df_coding.set_index(" ")

Culture_col = len(df_coding["CULTURE"].columns)
Event_num = 6
Cause_num = 9
Action_num = 9
Other_num = 1
#%% md
# <font size = 5px>Now run the [Create cleaned and refitted culture cells for later coding](#create-cleaned-and-refitted-culture-cells-for-later-coding) cell</font>
#%% md
# #### <br><br>(OPTONAL) Create cleaned Output Excels
# 
#%% md
# _listOfInclusions.xlsx
#%%
assert useDatasets==True, "You do not have dataset indicators loaded"

# CHANGE: enter in the key for the dictionary that contains your datasets (usually is '1' and '2')
ds_ref = '1' # this dataset will be used as the reference that the other dataset will try to match in its count
ds_tar = '2' 

# Create a column for if we should include the passage or not, then include all passages in the reference dataset
df['Include'] = 0
df.loc[df['Passage Number'].isin(df_coding[('CULTURE','Passage Number')]), 'Include'] = 1
# Create Include list from the base dataframe. the count should be 
df_inclusionList = df.copy()
df_inclusionList = df_inclusionList[['Passage Number','Dataset', 'Include']]


assert sum(df_include['Include'])==11601, "Include count does not match original dataframe for this dataset. Please be careful you are not running this code when you shouldn't"
# df_Freqs_updated.to_excel(directory+'/Cleaned_Cultures/'+'_Culture_Split_Count.xlsx')
df_inclusionList.to_excel(directory+'/Cleaned_Cultures/'+'_listOfInclusions.xlsx')
#%% md
# _Culture_Split_Count.xlsx
#%%
df_OCMExplode = df.loc[df['Include']==1].explode(column='OCM').reset_index(drop=True)
#get value count for all OCMs regardless of its inclusion
valcount_OCMCultures = df_OCMExplode.value_counts(subset=['Culture','OCM'])

cultures = valcount_OCMCultures.index.get_level_values('Culture').unique()
df_Freqs = pd.DataFrame(0,index=cultures, columns=["ReferenceDS_Freq","TargetDS_Freq_Raw", "TargetDS_Freq_Redux"]+scrapedOCMs)

df_Freqs['ReferenceDS_Freq'] = df.loc[df['Dataset']=='1'].value_counts(subset='Culture', ascending=True) #Get Reference number
df_Freqs['TargetDS_Freq_Raw'] = df.loc[df['Dataset']=='2'].value_counts(subset='Culture', ascending=True) #Get Target number before reduction
df_Freqs['TargetDS_Freq_Redux'] = df.loc[(df['Dataset']=='2') & (df['Include']==1)].value_counts(subset='Culture', ascending=True) #Get Target number after reduction
df_Freqs[scrapedOCMs] = valcount_OCMCultures.unstack(fill_value=0)[scrapedOCMs] # Get OCM counts after reduction for all datasets (unstack creates a dataframe from multiindexes)
df_Freqs.head(4)

df_Freqs.to_excel(directory+'/Cleaned_Cultures/'+'_Culture_Split_Count.xlsx')
#%% md
# _CultureCounts.xlsx
#%%
# create culture counts dataframe
# df_cultCount = df.loc[df["Include"]==1].copy() #for this emergency, df must be chnaged
df_culture = pd.DataFrame({"OWC":[], "Region":[], "Culture":[], "Partition":[], "Total Passage Count":[], "Partitioned Passage Count":[]})
df_culture[["Culture", 'Partition', "Total Passage Count", "Partitioned Passage Count"]] = pd.concat([df_partition['Culture'],  df_partition['Partition_Num'],  df_partition['Total_size'], df_partition['Partition_Size']],axis=1)

df_reduced = df.loc[df["Include"]==1, ["OWC", "Culture", "Region"]].drop_duplicates()
for i, rows in df_partition.iterrows():
    culture = rows["Culture"]
    OWC = df_reduced.loc[df_reduced["Culture"]==culture].iloc[0]["OWC"]
    Region = df_reduced.loc[df_reduced["Culture"]==culture].iloc[0]["Region"]
    df_culture.loc[i, ["OWC","Region"]] = [OWC, Region]
df_culture.sort_values(by = "Total Passage Count")
#%%
# Save culture counts
df_culture = df_culture.sort_values(by="Total Passage Count")
df_culture.to_excel(path+"/_CultureCounts.xlsx", index=False)

#%% md
# #### <br><br>(DELETE) check to make sure it worked using the temporary old files
#%%
directory = "../Data/RA coded files (temporary, you may delete)/"
df_orig = cultFileExtract(directory, False)

# Drop empty first column and empty first row(s)
df_orig = df_orig.drop(columns=df_orig.columns[0])
df_orig = df_orig.dropna(subset=[("CULTURE","Passage Number")])
#OCM strip
df_orig = OCM_stripper(df_orig, OCM=('CULTURE','OCM')); 
df_orig= df_orig.astype({('CULTURE','Passage Number'): 'int32'})
#%%
df_orig = df_orig.drop(columns=dropcolumns, axis=1, level=0) #The "axis" and "level" Parameters are not necessary in this case but help speeds (and get rid of warnings) when using more complicated dropping like dropping by multiindex not multiheader
passage_list = set(df_orig[("CULTURE", 'Passage Number')])
#%%
# df_orig.loc[df_orig[("CULTURE", 'Passage Number')] is in ["37222", "37223"]]
df_codingRdx = df_coding[df_coding[("CULTURE", 'Passage Number')].isin(passage_list)]
df_codingRdx = df_codingRdx.sort_values(by = ("CULTURE", 'Passage Number'), ascending=True)
df_orig = df_orig.sort_values(by = ("CULTURE", 'Passage Number'), ascending=True)
df_codingRdx.head(2)
#%%
print('Incongruent columns:',sum(df_orig.columns != df_codingRdx.columns))
#%%

df_codingRdx = df_codingRdx.set_index(('CULTURE','Passage Number'))
df_codingRdx.index.name = "Passage Number"
df_orig = df_orig.set_index(('CULTURE','Passage Number'))
df_orig.index.name = "Passage Number"
#%%

failureTrigger = False
for column in df_codingRdx.columns:
    
    match = df_codingRdx[column].equals(df_orig[column])
    if match is False:
        mismatches = [a != b if not all([pd.isna(a),pd.isna(b)]) else False for a, b in zip(df_codingRdx[column], df_orig[column]) ] # check if there is a mismatch between columns so long as both valeus are not NaN (which will always mismatch)
        print(f"{sum(mismatches)} incongruities in column {column}")
        failureTrigger = True
if failureTrigger == False:
    print("All values match")
#%%


failureTrigger = False
for column in df_codingRdx.columns:
    orig = df_orig[column].dropna()
    redux = df_codingRdx[column].dropna()
    match = sum(orig != redux)
    if match >0:
        # mismatches = [a != b for a, b in zip(orig, redux) ] # check if there is a mismatch between columns so long as both valeus are not NaN (which will always mismatch)
        print(f"{match} incongruities in column {column}")
        failureTrigger = True
if failureTrigger == False:
    print("All values match")
#%%
msk = [a != b if not all([pd.isna(a),pd.isna(b)]) else False for a, b in zip(df_codingRdx[('EVENT',"No_Info")], df_orig[('EVENT',"No_Info")]) ]
df_compare = pd.DataFrame({'Original':df_orig[('EVENT',"No_Info")], 'Redux':df_codingRdx[('EVENT',"No_Info")]})
df_compare[msk]