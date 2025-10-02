#%%
import pandas as pd
import os
import numpy as np
import nltk
import openpyxl
#%% md
# # RA Code Cleaning and Exploration
#%% md
# ## Cleaning
#%% md
# This notebook is meant to combine individual culture codings from the research assistants and do small exploratory analysis on them.<br>
# Many errors with the RA coding are meant to be manually corrected by coders meaning ideally whenever an issue is found, it is corrected and these scripts are ran until no errors are found or such errors are able to be corrected progamatically through the script (and don't need coder intervention)
#%%
# Get all cultural files for specified directory
def cultFileExtract(directory, finished:bool=False) -> pd.DataFrame:
    # Determine if path even exists
    df_coding = pd.DataFrame()
    if os.path.exists(directory) == False:
        return None

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file worth extracting (disregard temp files, recodes, Eric's files, and non-xlsx files)
        if all([os.path.isfile(f), filename.endswith(".xlsx")]) and not any([filename.startswith("_"), filename.__contains__("Eric"), filename.__contains__("recode"), filename.__contains__("Errors"), filename.startswith("~$")]):
            df_coding_culture = pd.read_excel(f, header=[0,1])
            # Mark if 1st of 2nd run
            if f.__contains__("_2ndRun"):
                df_coding_culture[("CODER","Run_Number")] = 2
            else:
                df_coding_culture[("CODER","Run_Number")] = 1
            df_coding = pd.concat([df_coding, df_coding_culture], ignore_index=True)
            print(f"Used: {f}")
    if len(df_coding) > 0: #if any file was loaded, mark True or ƒalse for finished
        df_coding[("CODER","Finished")] = finished
    return df_coding.copy()

#%%
# Load and Append dataframe for all finished cultures by RA's
# CHANGE each of these to fit the location of the RA coded cultures as well as where you want these files to end up (preferably the same location as the _altogether_Dataset they originated from)
# Otherwise Uncomment

# # Sickness dataset (OCMs 750-753)
# RA_list = ["AH","HL","KK","KY","LG","YM"] # RAs Spring 2023
# folder = r'subjects-(sickness)_FILTERS-culture_level_samples(PSF)'                          # OCM 750
# OutputDir = '../Data/' + folder
# RACodingDir = OutputDir +'/RA Codings/'

# Sickness (OCMs 750-753) and nonsickness dataset (OCMs '784, '731', '732', '777', '791', '793')
RA_list = ["AH","CHD","JM","KB","KY","LR","MJ","SC"] # RAs Fall 2023
RA_list = RA_list + ["JH","BK","AL","SJ","KW"] # RAs Spring 2024 (appended to Fall, "JM","KB","LR" came back for spring)

RACodingDir = '../RA Coding/Coding/'
folder = r'(subjects-(contracts_OR_disabilities_OR_disasters_OR_friendships_OR_gift_giving_OR_infant_feeding_OR_lineages_OR_etc'   # OCM ['586' , '684' , '688' , '731' , '732' , '756' , '767' , '777' , '791' , '792' , '793' , '431' , '572' , '594' , '613' , '624' , '675' , '853']
OutputDir = '../Data/' + folder


df = pd.DataFrame()
warning_list = []
for RA in RA_list:
    # Extract Unfinished 
    directory = RACodingDir + RA
    df_RA_unfin = cultFileExtract(directory, finished=False)
    # Extract Finished
    directory += '/Finished'
    df_RA_fin = cultFileExtract(directory, finished=True)

    # concatinate the finished and unfinished then set the column by name
    df_RA_fin = pd.concat([df_RA_fin, df_RA_unfin], ignore_index=True)
    if len(df_RA_fin) == 0:
        warning_list.append(RA)
        continue
    df_RA_fin[("CODER","Coder")] = RA
    df = pd.concat([df, df_RA_fin], ignore_index=True)


# Drop empty first column and empty first row(s)
df = df.drop(columns=df.columns[0])
df = df.dropna(subset=[("CULTURE","Passage Number")])
print(f"Unfinished Cultures: {len(set(df[~df[('CODER','Finished')]][('CULTURE','Culture')]))}")
print(f"Finished Cultures: {len(set(df[df[('CODER','Finished')]][('CULTURE','Culture')]))}")
print(f"DATAFRAME ROWS: {len(df)}")
if len(warning_list) != 0:
    print('\033[93m'+ f"WARNING, no files found for {warning_list}"+ '\033[0m')
#%%
# # Quick Passage search 
# df.loc[df[('CULTURE','Passage Number')]==3689]

# # Quick Culture by RA search
# RA = 'KW'
# Culture = 'Eastern Toraja'
# print(len(df.loc[(df[('CULTURE','Culture')]==Culture)&(df[('CODER','Coder')]==RA)]))
# print(cult_dict[(Culture,RA)])
#%%
# Get Cultural dictionary
cult_dict = dict(df[[('CULTURE','Culture'),('CODER','Coder')]].value_counts())
print(f"DF:       {sum(df[('CULTURE','Culture')].value_counts())}")
print(f"CultDict: {sum([val for key, val in cult_dict.items()])}") 

#%%
# strip OCMs and make passage an integer
import re
def OCM_stripper(df, OCM='OCM'):
    if type(df[OCM].iloc[0]) is list: # if already a list, return without alteration
        return df
    df_ocm = df.copy() # so that the original dataframe is not affected
    df_ocm[OCM] = df_ocm[OCM].apply(lambda x: re.sub(" |\'",'',x))
    df_ocm[OCM] = df_ocm[OCM].apply(lambda x: x[1:-1].split(','))
    return df_ocm
#%%
df = OCM_stripper(df, OCM=('CULTURE','OCM')); 
df= df.astype({('CULTURE','Passage Number'): 'int32'})
#%% md
# ### Remove blank and duplicate passages
#%% md
# NOTE as of writing this, there is a duplicate passage (48602 and 48610) which appears because the RA may have copied and pasted 48602 into 48610. It appears this 48610 was coded using the new text and thus constitutes a true duplicate. Although the passage is correctly later, we should remove this passage rather than fixing it (since the codings would not be representative of the new passage). If you see that there are ANY duplicates of passage numbers or text (which I have not described here), be wary as this should not normally be possible without some RA error.
#%%
# Check No_info for counts
df[("EVENT","No_Info")].value_counts(dropna=False)
#%%
#remove "/" and NaN rows
df = df[df[("EVENT","No_Info")] != "/"]
# remove all rows which have empty "No_Info" Note that some of these rows were recoded by EC if they had partial information enough to glean. Check comments
df = df.dropna(subset=[("EVENT","No_Info"), ("CAUSE","No_Info"), ("ACTION","No_Info")], how='all')  
df[("EVENT","No_Info")].value_counts(dropna=False)
#%%
# Any duplicate passage or passage numbers remaining?
dups_num = df.duplicated(("CULTURE","Passage Number"))
if sum(dups_num) >0:
    cultureDups = set(df[dups_num][("CULTURE","Culture")])
    print('\033[93m'+ f"WARNING, {sum(dups_num)} passage NUMBER duplicate(s) found between the following cultures: {cultureDups}"+ '\033[0m')
else:
    print("No passage number duplicates")
dups_text = df.duplicated(("CULTURE","Passage"))
if sum(dups_text) >0:
    cultureDups = set(df[dups_text][("CULTURE","Culture")])
    print('\033[93m'+ f"WARNING, {sum(dups_text)} passage TEXT duplicate(s) found between the following cultures: {cultureDups}"+ '\033[0m')
else:
    print("No passage text duplicates")
#%%
# Remove Passage Number and Text duplicates
print("Before",len(df))
df = df[(~dups_text) & (~dups_num)]
print("After",len(df))
#%%
# Any duplicate passage or passage numbers remaining?
dups_num = df.duplicated(("CULTURE","Passage Number"))
if sum(dups_num) >0:
    cultureDups = set(df[dups_num][("CULTURE","Culture")])
    print('\033[93m'+ f"WARNING, {sum(dups_num)} passage NUMBER duplicate(s) found between the following cultures: {cultureDups}"+ '\033[0m')
else:
    print("No passage number duplicates")
dups_text = df.duplicated(("CULTURE","Passage"))
if sum(dups_text) >0:
    cultureDups = set(df[dups_text][("CULTURE","Culture")])
    print('\033[93m'+ f"WARNING, {sum(dups_text)} passage TEXT duplicate(s) found between the following cultures: {cultureDups}"+ '\033[0m')
else:
    print("No passage text duplicates")
#%% md
# ### Check for missing or wrong codings
#%%
multiCol = list(df.columns)
valuecountCol = []
for col in multiCol:
    if col[0] in ["CULTURE", "OTHER", "CODER"] or col[1] in ["Description", "Local_terms", "Local_Terms"]:
        continue
    else:
        valuecountCol.append(col)

# look through and make sure each column only has a 1 or 0
for col in valuecountCol:
    Counts = df[col].value_counts(dropna=False)
    if len(Counts) >2:
        print(Counts)
    else:
        print(f"{col}  {(45 - len(str(col))) * ' '}CORRECTLY contains ONLY 2 values")

#%%
# # Save mistakes in a dataframe and check
# df_mistakes = pd.DataFrame()
# # cycle through all columns and select all rows which do not contain 0 or 1
# for col in valuecountCol:
#     df_mistakesRA = df[(df[col] != 1) & (df[col] != 0)]
#     df_mistakes = pd.concat([df_mistakes, df_mistakesRA], ignore_index=True) 
# mistake_list = list(set(df_mistakes[('CULTURE','Passage Number')]))
# # Remove bad rows from dataset
# df = df[~df[('CULTURE','Passage Number')].isin(mistake_list)]
# df[valuecountCol] = df[valuecountCol].astype(int)
# df_mistakes.sort_values(by=[("CODER","Coder"), ("CULTURE","Passage Number")], ascending=True)[ [("CULTURE","Culture"), ("CULTURE","Passage Number")]+ valuecountCol + [("CODER","Coder")]]
#%% md
# ## RA Error cleaning
#%% md
# ### Find and Remove Binary Errors
#%%
def mistakeResp(df=df):# Get Binary errors:
    header = pd.MultiIndex.from_tuples(valuecountCol)
    df_binaryErrors = pd.DataFrame(columns=header, index=df.index, data=np.zeros(df[valuecountCol].shape)) # create filled 0's dataframe for replacing with mistake counting

    # cycle through all columns and select all rows which do not contain 0 or 1
    for col in valuecountCol:
        binaryErrorsBool_series = (df[col] != 1) & (df[col] != 0)
        df_binaryErrors[col] = binaryErrorsBool_series.astype(int)
        df.loc[binaryErrorsBool_series, col] = np.nan #change the binary errors to NAN for later corrections
        
    # report mistakes
    mistake_list = df_binaryErrors.sum(axis=1).astype(bool) 
    # df = df[~df[('CULTURE','Passage Number')].isin(mistake_list)] ## Keeping this here because I like to remember this is an option (this specific one will not work for this new code)
    print("Mistakes:",len(df[mistake_list]))
    return df_binaryErrors, mistake_list

df_binaryErrors, mistake_list= mistakeResp(df)
#%% md
# ### Check for incongruous coding (No_info 1 while there is a 1 in another column) and pair with binary errors
#%%

topHeaders_list = ['EVENT', 'CAUSE', 'ACTION']
errorHeaders_list = ["Absent Errors", "Present Errors", "Binary Errors"]
errors_bool = [False]* len(df)
header = pd.MultiIndex.from_product([topHeaders_list, errorHeaders_list], names=['Category', 'Errors'])
df_errors = pd.DataFrame(columns=header)
df_errors_total = pd.DataFrame(columns=errorHeaders_list)


# concatinate into 
for topHeader in topHeaders_list:
    df_mainCat = df[topHeader]
    headers = [x for x in df_binaryErrors.columns if x[0] == topHeader]
    lowerHeaders = df_mainCat.columns[1:-2] # get subcategory coding column names without 'No_Info'
    assert [x[1] for x in headers][1:] == list(lowerHeaders), "Missing or mismatched columns"

    # sum accross rows of subcategories
    row_sum = df_mainCat.loc[:, lowerHeaders].sum(axis=1) 

    # Coding absent of info ( 1 for no_info) but there is a subcategory marked present
    absent_error = (df_mainCat['No_Info']==1) & (row_sum>0)
    # Coding present of info ( 0 for no_info) but there is no subcategory marked
    present_error = (df_mainCat['No_Info']==0) & (row_sum==0)

    # Get count of binary errors
    binary_error = df_binaryErrors[headers].sum(axis=1)

    errors_bool += absent_error+present_error+binary_error.astype(bool)+errors_bool
    # errors_bool += absent_error+present_error+errors_bool
    df_errors[[(topHeader, errorHeaders_list[0]),(topHeader, errorHeaders_list[1]),(topHeader, errorHeaders_list[2])]] = pd.concat([absent_error.astype(int), present_error.astype(int), binary_error], axis=1)
    df_errors_total.loc[topHeader] = [sum(absent_error), sum(present_error), sum(binary_error)]

df_errors_total
#%% md
# In the previouse 750 dataset, 20 values were corrected for typing errors and interpretated based on Research assistant descriptions and Research manager's final word. a good number were merely NaN values and were recoded as being 0's <br>
# Rows were also removed which have a missing no_info value as these tended to be completely blank <br>
# Codings found that were clearly wrong but did not violate coding rules (marking 1 for "Illness" as being present when it doesn't appear to be in the passage when we check) were not changed for the sake of not interfering with coder's reliability or decision.
#%%
# Print error rows
df_errors_shortened = pd.concat([df[errors_bool][[("CULTURE","Culture"), ("CULTURE","Passage Number"), ("CODER","Coder")]], df_errors[errors_bool]], axis=1)
df_errors_shortened
# df[errors_bool].sort_values(by=[ ("CODER","Coder"), ("CULTURE","Culture")], ascending=True)[ [("CULTURE","Culture"), ("CULTURE","Passage Number")]+ valuecountCol + [("CODER","Coder")]]
#%%
# OPTIONAL save correction file to RA's folder

RAs = set(df.loc[errors_bool, ("CODER","Coder")]) # get RAs that have wrong codings


for RA in RAs:
    df_RA = pd.DataFrame()
    RA_errors_bool = errors_bool & (df[("CODER","Coder")]==RA) # get error boolean into a single list for ease
    df_RA_intermediate = df.loc[RA_errors_bool]

    #Construct columns and sum up error
    df_RA[["Culture","Passage Number"]] = df_RA_intermediate[[("CULTURE","Culture"), ("CULTURE","Passage Number")]]
    for topHeader in topHeaders_list:
        df_RA[f"{topHeader} Errors"] = df_errors[topHeader][RA_errors_bool].sum(axis=1)
    df_RA["TOTAL"] = df_errors[RA_errors_bool].sum(axis=1)
    df_RA.loc[len(df_RA), "Culture"] = "PLEASE CORRECT THESE ERRORS THEN DELETE THE FILE WHEN YOU ARE FINISHED"
    # save to RA file
    df_RA.to_excel(RACodingDir+RA+"/Errors_To_Correct.xlsx", index=False)


#%% md
# #### Remove all errors from the dataframe
# 
#%%
df = df[~(mistake_list+errors_bool)]
df_binaryErrors, mistake_list= mistakeResp(df)
#%% md
# ### Fix Passage Text
#%% md
# There is a chance RAs have changed the passage text in some way (even by accident) so it is good to reset them
#%%
# Load original dataset
df_orig = pd.read_excel(OutputDir+"/_Altogether_Dataset_CLEANED.xlsx")
df_origCol = df_orig.columns[df_orig.columns.isin(df["CULTURE"])]
df_orig = df_orig.loc[:, df_origCol]
#%%
# Sort df by descending
df = df.sort_values(by =("CULTURE", "Passage Number"))
df_RA = df.copy()
df_RA = df_RA.loc[df_RA[("CODER", 'Run_Number')] == 1]["CULTURE"] #only use first run
# df_RA.at[len(df_RA)-1, 'Passage Number'] = -9

pass_list = list(df_RA["Passage Number"])
pass_list.sort() #sorting may not be needed but I am going to do it anyway

# # get only the original passages shared with those coded by RAs
assert len(set(df_RA["Passage Number"]).difference(set(df_orig["Passage Number"]))) == 0, "Error, Some passages in ther RA df are not in the original"
df_orig = df_orig.loc[df_orig["Passage Number"].isin(pass_list)]
dataLen = len(df_orig)
assert dataLen == len(df_RA), "lengths are incongruous and do not match up"

alreadyPresentDups = sum(df_orig.duplicated(subset='Passage Number'))
print(f"Duplicate Passage Numbers in the original dataset: {alreadyPresentDups}")
print(f"Duplicate Passages in the original dataset: {sum(df_orig.duplicated(subset='Passage'))}")
print(f"Duplicate Passage Numbers in the RA dataset: {sum(df_RA.duplicated(subset='Passage Number'))}")
print(f"Duplicate Passages in the RA dataset: {sum(df_RA.duplicated(subset='Passage'))}")

df_merged = pd.concat([df_RA, df_orig])
incong_bool = df_merged.duplicated(subset=['Passage Number','Passage'], keep=False)
print(f"Incongruent passages between datasets (not subtracting already present dups): {(len(df_RA)+ len(df_orig) - sum(incong_bool))/2}")

#%%
#Investiagte the Incongruent Passages
for pasNumber in df_merged[~incong_bool]["Passage Number"].unique():
    print("\nPassage Number:",pasNumber, "     Culture:", df_merged[df_merged["Passage Number"]==pasNumber]["Culture"].iloc[0], "       RA:", df[df[("CULTURE","Passage Number")]==pasNumber][("CODER", "Coder")].iloc[0])

    pas_orig = df_orig[df_orig["Passage Number"]==pasNumber]["Passage"]
    assert len(pas_orig) ==1, f"ERROR, {len(pas_orig)} found"
    print("Original: ", pas_orig.iloc[0])

    pas_RA = df_RA[df_RA["Passage Number"]==pasNumber]["Passage"]
    assert len(pas_RA) ==1, f"ERROR, {len(pas_RA)} found"
    print("RA COded: ", pas_RA.iloc[0])


# print(df_merged[df_merged["Passage Number"]==42467]["Passage"].iloc[0])
# print(df_merged[df_merged["Passage Number"]==42467]["Passage"].iloc[1])
# df_merged
#%%
# # replace RA Passages with original passages (this time we can include all runs)
df = df.reset_index(drop=True)
df_orig = df_orig.reset_index(drop=True) #first reset both indexes
df_RA = df["CULTURE"].copy()



merged_df = pd.merge(df_RA, df_orig[['Passage Number', 'Passage']], on='Passage Number', suffixes=('_dummy', '_orig'))

# Replace 'Passage' in df_dummy with 'Passage_orig'
df[('CULTURE','Passage')] = merged_df['Passage_orig']
df_RA = df['CULTURE'].copy()


dataLen = len(df_RA)
df_merged = pd.concat([df_RA, df_orig])
incong_bool = df_merged.duplicated(subset=['Passage Number','Passage'], keep=False)
print(f"Incongruent passages between datasets (not subtracting already present dups): {(len(df_RA)+ len(df_orig) - sum(incong_bool))/2}")
#%%
~(df[('CULTURE','Passage Number')]==90450)
#%%
# Remove passages which are changed and are deemed as changing the interpretition and affecting an RAs coding.
# Enter in Passage ID as you see fit NOTE That these IDs below may be different if you are not using the sickness + non-sickness datasets
passage_nums = [90450]
print("Before: ", len(df))
for passage_num in passage_nums:
    df = df[~(df[('CULTURE','Passage Number')]==passage_num)]
print("After: ", len(df))
#%% md
# ## Save Cleaned File
#%% md
# ### Add Dataset Indicators (optional if you have multiple datasets within a run of RA coding)
#%%
try:
    df_datasets = pd.read_excel(f"{OutputDir}/_Dataset_Lists.xlsx")
    df_dummy = df["CULTURE"].copy()
    merged_df = pd.merge(df_dummy, df_datasets[['Passage Number','Dataset']], on='Passage Number', suffixes=('_1and2', '_2dataset'), how='left')
    print(sum(merged_df["Dataset"].isna()), "Missing dataset indicators")

    df[('CODER', "Dataset")] = merged_df["Dataset"]
except:
    print("No datasets loaded")
#%% md
# ### Save
#%%
# Save concatinated file
df.to_excel(OutputDir+"/_Altogether_Dataset_RACoded.xlsx")
#%%
# DELETE
# duplicate count based on multiple columns of meta data but not using passage number or contents

#load first dataset of sickness
folder = r'subjects-(sickness)_FILTERS-culture_level_samples(PSF)'                          # OCM 750
Dir = '../Data/' + folder + "/_Altogether_Dataset_RACoded.xlsx"
df_2 = pd.read_excel(Dir, header=[0,1], index_col=0)
df_2 = OCM_stripper(df_2, OCM=('CULTURE','OCM'))


df_dummy = df_2["CULTURE"].copy()

dropMetaColumn_list = ['Passage Number','Passage']
df_dummy = df_dummy.drop(columns=dropMetaColumn_list)
#sort OCMs before we turn them into a string (to check duplicates)
df_dummy["OCM"].apply(lambda x: x.sort()) 
df_dummy["OCM"] = df_dummy["OCM"].apply(lambda x: str(x))


print(f"Duplicates by Passage Number: {sum(df_2.duplicated(subset=('CULTURE','Passage Number')))}")
print(f"Duplicates by Passage: {sum(df_2.duplicated(subset=('CULTURE','Passage')))}")
print(f"Duplicates by MetaData: {sum(df_dummy.duplicated())}")

#%% md
# ## (Optional) RA Response Accuracy and Progress
# 
# 
#%% md
# ### RA Progress
#%%
print(cult_dict["Azande","KB"])
df.loc[df[("CULTURE",'Culture')]=="Azande"].head(2)
#%%
# def space(*args):
#     spacerString = ""
#     for arg in args:
#         assert (type(arg) is tuple) and (type(arg[0]) is str) and (type(arg[1]) is int), "Must be a tuple of string followed by a desired spacer, you may alo"
#         spaces = ' '*(arg[1]-len(arg[0]))
#         if len(arg) == 2 or arg[2] == "Before":
#             spacerString += spaces+arg[0]
#         elif len(arg) == 3 and arg[2] == "After":
#             spacerString += arg[0]+spaces
#         else:
#             raise Exception("Invalid format, must be a tuple of string, then Int then optional string of Before or After")
#     return spacerString
        
#%%
# for spacing, give tuples () of the string, number of spaces, followed by "Before" or "After" to indicate where the spacers go (optional, default is After)
def spacer(*args):
    spacerString = ""
    for arg in args:
        assert (type(arg) is tuple) and (type(arg[0]) is str) and (type(arg[1]) is int), "Must be a tuple of string followed by a desired spacer, you may alo"
        spaces = ' '*(arg[1]-len(arg[0]))
        if len(arg) == 2 or arg[2] == "After":
            spacerString += arg[0]+spaces
        elif len(arg) == 3 and arg[2] == "Before":
            spacerString += spaces+arg[0]
        else:
            raise Exception("Invalid format, must be a tuple of string, then Int then optional string of Before or After")
    return spacerString


def num_create():
    number = str(cult_dict[(cult, RA)] - len(df_CultDummy))
    number = f"{spacer((number,5,'Before'))} PL"
    return number


#Percentage of how much a culture is complete
df = df.sort_values(by=[("CODER","Coder")]) # sort so the list can be by RA
results = zip(df[("CULTURE","Culture")], df[("CODER","Coder")])
results = sorted(list(set(results)), key=lambda x: x[1])
fins_str = ""
unfins_str = ""
notstart_str = ""
cultCount_dict = {"Finished":0,"Unfinished":0,"Not Started":0} # count number of cultures for each category
# display culture progress by RAs
for cult, RA in results:
    if cult == "Azande":
        pass
    df_CultDummy = df.loc[(df[("CULTURE","Culture")] == cult) & (df[("CODER","Coder")]==RA)]
    percentage = str(int((len(df_CultDummy) / (cult_dict[(cult, RA)]))*100))
    percentage = (' ' * (4- len(percentage))) + percentage + '%'
    number = str(cult_dict[(cult, RA)] - len(df_CultDummy))
    number = f"{spacer((number,5,'Before'))} PL"
    #If finished, put it in the first paragraph, otherwise put it in the second
    if df_CultDummy[("CODER","Finished")].iloc[0]: 
        finished = "Finished"
        fins_str += f"{spacer((RA,6), (cult,20), (percentage,4))}{number}    {finished}\n"
        # fins_str += f"{RA}{' '*(6-len(RA))}{cult}{' '*(20-len(cult))}{percentage}{' '*(4-len(percentage))}{number}    {finished}\n"
    else:
        finished = "Unfinished"
        unfins_str += f"{spacer((RA,6), (cult,20), (percentage,4))}{number}    {finished}\n"
    cultCount_dict[finished] += 1
# display cultures not started
for cult, RA in set(cult_dict.keys()) - set(results):
    percentage = "   0%"
    finished = "Not Started"
    number = str(cult_dict[(cult, RA)])
    number = f"{spacer((number,5,'Before'))} PL"
    notstart_str += f"{spacer((RA,6), (cult,20), (percentage,4))}{number}    {finished}\n"
    cultCount_dict[finished] += 1
fins_str = f"-------FINISHED-------{' '*4}  Cultures: {cultCount_dict['Finished']}\n{fins_str}\n\n"
unfins_str = f"-------UNFINISHED-------{' '*2}  Cultures: {cultCount_dict['Unfinished']}\n{unfins_str}\n\n"
notstart_str = f"-------NOT STARTED-------{' '*1}  Cultures: {cultCount_dict['Not Started']}\n{notstart_str}\n\n"
total = sum(value for value in cultCount_dict.values())
print(f"Total:{total}\n\n{fins_str}{unfins_str}{notstart_str}\n\n PL = Passages Left to code")
#%%
from datetime import date
today = date.today() #get date


# Passages per hour list (Includes finished and unfinished)
PPH_list = []
df_RA = pd.DataFrame({"RA":[], "Passages":[], "Hours":[], 'PassPerHour':[], 'Words':[], 'WordsPerHour':[], 'Date':[]})
# hours_dict = {'LG':48.5, 'AH':52.1, 'KK':27.9, 'HL':57.2, 'YM':97.6, 'KY':47.3}
hours_dict = {'LR':132.2, 'KB':93.5, 'JM':126.8, 'JH':45.8, 'BK':67.2, 'AL':3.5, 'SJ':64.0, 'KW':46.6, 'MJ':42.4, 'SC':44.9,  'KY':6.6, 'AH':48.7, 'CHD':61.6}
code_count = df[("CODER","Coder")].value_counts()
for key, val in hours_dict.items():
    PPH = round(code_count[key]/val, 2)
    PPH_list.append(PPH)
    df_word_Count = df.loc[df[("CODER","Coder")]==key]
    word_count = df_word_Count[("CULTURE","Passage")].str.split().str.len().sum()
    WPH = round(word_count/hours_dict[key], 2)
    print(f"{key}{(4-len(key))*' '}{code_count[key]}{(10-len(str(code_count[key]))-len(str(PPH)))*' '}{PPH}   Passages per hour         {word_count}{(8-len(str(word_count)))*' '}{(8-len(str(WPH)))*' '}{WPH}  Words Per Hour")
    df_RA.loc[len(df_RA.index)] = [key, code_count[key], val, PPH, word_count, WPH, today ]
print(f"mean speed {round(np.mean(PPH_list),2)} Passages per hour")
#%%
# Save Progress

fileName = "RA_Progress.xlsx"


if os.path.isfile(fileName):
    df_RA_old = pd.read_excel(fileName, index_col=False)
    df_RA = pd.concat([df_RA, df_RA_old], ignore_index=True)
    df_RA = df_RA.astype({'Date':'datetime64[ns]'})
    df_RA = df_RA.sort_values(by=['Date','RA'], ascending=[True,False])
    df_RA = df_RA.drop_duplicates(subset=["RA","Date"], keep="last") #may work to allow updating the old rows, not sure, haven't tested
    df_RA = df_RA.reset_index(drop=True)
df_RA.to_excel(fileName, index=False)

#%%
# Counts
print(f"Unfinished but started (unique) Cultures: {len(set(df[~df[('CODER','Finished')]][('CULTURE','Culture')]))}")
print(f"Unfinished Culture Passage count: {len(df[~df[('CODER','Finished')]][('CULTURE','Culture')])}\n")
print(f"Finished (unique) Cultures: {len(set(df[df[('CODER','Finished')]][('CULTURE','Culture')]))}")
print(f"Finished Culture Passage Count: {len(df[df[('CODER','Finished')]][('CULTURE','Culture')])}\n")
print(f"Total (unique) Cultures: {len(set(df[('CULTURE','Culture')]))}")
print(f"Total Passages: {len(df)}")
#%%
print(f"1st Run Cultures: {len(set(df[df[('CODER','Run_Number')]==1][('CULTURE','Culture')]))}")
print(f"1st Run Culture Passage Count: {len(df[df[('CODER','Run_Number')]==1][('CULTURE','Culture')])}\n")
print(f"2nd Run Cultures: {len(set(df[df[('CODER','Run_Number')]==2][('CULTURE','Culture')]))}")
print(f"2nd Run Culture Passage Count: {len(df[df[('CODER','Run_Number')]==2][('CULTURE','Culture')])}\n")
#%% md
# ### Inter-Rater Reliability Coding
#%% md
# Check agreement between coders
#%%
# Cultures
df_one = df.loc[df[("CODER","Run_Number")]==1]
print(df_one[("CULTURE","Culture")].unique())
df_two = df.loc[df[("CODER","Run_Number")]==2]
print(df_two[("CULTURE","Culture")].unique())

#%%
df_CatFiltered = df[df.duplicated(subset=("CULTURE","Passage Number"), keep=False)].copy()

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
#%%
df_one = df_CatFiltered.loc[df_CatFiltered[("CODER","Run_Number")]==1]
df_two = df_CatFiltered.loc[df_CatFiltered[("CODER","Run_Number")]==2]

df_comparison = df_one == df_two
df_comparison.mean(axis=0)


#%% md
# ## (Optional) Explore Classification Bias
#%% md
# ### Response bias by class
#%%
1-np.mean(df[valuecountCol[1]])
#%%
# # Calculate response bias (unbalanced 1 vs. 0, a perfectly balanced one would be .5 or 50%)
# def colProportion(df, col, present=1):
#     value_counts = df[col].value_counts()
#     proportion = round(value_counts[present]/len(df),2)
#     return proportion

# set up dataframe for easy saving
df_biases = pd.DataFrame(columns=["Class","Raw_Bias","Adj__Bias"])

# Get proportions and show table. 'raw' is just number of present divided by total while 'adj' is within main category proportion present divided by total main class present
print("BIAS FOR ANSWERING \'PRESENT\'")
print("Passage Count: ", len(df))
print(f"{' '*39}Raw{' '*10}Adj.") 
print(f"{'_'*60}")
for col in valuecountCol:
    if col[1] == "No_Info":
        proportion = 1-np.mean(df[col])
        mainCat_proportion = proportion
        proportion = round(proportion,2)
        df_biases = pd.concat([df_biases, pd.DataFrame({"Class":[col[0]],"Raw_Bias":[proportion]})])
        print(f"\n{col[0]}:{(38-len(col[0]))*' '}{proportion}")
    else:
        proportion = np.mean(df[col])
        adj_proportion = round(proportion / mainCat_proportion,2) # get adjusted proportion within category
        proportion = round(proportion,2)
        df_biases = pd.concat([df_biases, pd.DataFrame({"Class":[col[1]],"Raw_Bias":[proportion], "Adj__Bias":[adj_proportion]})])
        print(f"\t{col[1]}:{(30-len(col[1]))*' '}{proportion}{' '*(12-len(str(proportion)))}{adj_proportion}")

# save biases
df_biases.to_excel(OutputDir+"/_Class_Biases.xlsx", index=False)
#%% md
# ### Classification bias by OCM
#%% md
# The idea is to see if there are certian OCMs that naturally are more associated with misfortune than others
#%%
import re
def OCM_stripper(df, OCM='OCM'):
    df[OCM] = df[OCM].apply(lambda x: re.sub(" |\'",'',x))
    df[OCM] = df[OCM].apply(lambda x: x[1:-1].split(','))
    return df
#%%
# CHANGE per main category and sub category
main_cat = "ACTION"
category = "Other"


non_list = ['No_Info',  'Local_Terms', 'Description', category]
filter_columns = [(main_cat, col) for col in list(df[main_cat].columns) if col not in non_list]
print("Filtered columns:", filter_columns,"\n")

# filter away all columns so that only the desired column is left
df_CatFiltered = df.copy()
for filter in filter_columns:
    df_CatFiltered = df_CatFiltered.loc[df_CatFiltered[filter]==0]
df_CatFiltered = df_CatFiltered.loc[df_CatFiltered[(main_cat, category)]==1]
df_CatFiltered = df_CatFiltered.explode(column=('CULTURE','OCM')).reset_index(drop=True) # explode the OCMs so they become their own row.

# set up Value counts dataframe
value_counts_df = df_CatFiltered[('CULTURE','OCM')].value_counts().reset_index()
value_counts_df.columns = ['OCM', 'Count']
value_counts_df['OCM'] = value_counts_df['OCM'].astype('int')
value_counts_df = value_counts_df[value_counts_df['Count'] >= 5] # remove rows that are below threshold
value_counts_df.set_index('OCM', inplace=True)
value_counts_df.head(7)

# combine value counts with OCM meaning
df_OCMlist = pd.read_excel("../../eHRAF_Scraper/Resources/OCM_Codes.xlsx", index_col="OCM")
value_counts_df = value_counts_df.merge(df_OCMlist, left_index=True, right_index=True, how='left')
print(f"Common OCMs for sub category {category} in main category {main_cat}")

value_counts_df[['Count','Meaning']]

#%% md
# ### Classification bias by passage length
#%%
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#%%
# get tokens of each passage then get the length (in log digits)
tokenized_length = df[('CULTURE','Passage')].apply(lambda x: np.log(len(tokenizer(x)['input_ids'])))

plt.hist(tokenized_length)

#%%
df_token_length = df.copy()
df_token_length = df_token_length[valuecountCol]
df_token_length[('CULTURE','tokenized_length')] = tokenized_length
#%%
import sys


print(f"{' '*39}Beta{' '*7}P-Val") 
print(f"{'_'*60}")
for col in valuecountCol:
    # Create small dataset of predictors and targets for easy indexing
    data = pd.DataFrame({
    "predictor_variable": df_token_length[('CULTURE','tokenized_length')],
    "target_variable":  df_token_length[col].astype(int)
    })
    x = data["predictor_variable"]
    x = data["target_variable"]

    if col[1] == 'No_Info':
        x = x.replace({0:1, 1:0})
    #     name = col[0]
    # else:
    #     name = f'\t{col[1]}'
    x = sm.add_constant(x) # add intercept
    model = sm.Logit(x, x)

    original_stdout = sys.stdout  # Save the original standard output
    sys.stdout = open('nul', 'w')  # Redirect standard output to a null device

    result = model.fit()

    # Restore the original standard output
    sys.stdout = original_stdout


    p_val = round(result.pvalues["predictor_variable"],3)
    beta = round(result.params["predictor_variable"],3)
    if col[1] == 'No_Info':
        print(f"\n{col[0]}:{(38-len(col[0]))*' '}{beta}{(11-len(str(beta)))*' '}{p_val}")
    else:
        print(f"\t{col[1]}:{(30-len(col[1]))*' '}{beta}{(11-len(str(beta)))*' '}{p_val}")

#%% md
# ### Redo old cultural coding
#%%
# a way to redo 
#%% md
# ## (Optional) Explore Dataset Distribution
#%% md
# ### By RA Finished cultures
#%%
# Check to see if the datasets are congregated in certain portions of the corpus. Ideally, the two dataset numbers should be equal
df_dataCheck = df.loc[df[("CODER","Finished")]].copy() # Only get finished cultures from DF
cultures = df_dataCheck[("CULTURE","Culture")].unique()
df_dataCheck[("CODER","Percentage")] = float()

# Add a column for "percentages" which says which percentage of the way through a culture did a passage appear
for culture in cultures:
    mask = df_dataCheck[("CULTURE","Culture")]==culture
    data_len = len(df_dataCheck.loc[mask])
    percRange = np.arange(1,data_len+1)
    percRange = percRange / data_len
    df_dataCheck.loc[mask, ("CODER","Percentage")] = percRange

# Report for each dataset the average placement of the passages (ideally should be around 50% for all of them)
for dataIndc in df_dataCheck[('CODER','Dataset')].unique():
    perc_ds = df_dataCheck.loc[df_dataCheck[('CODER','Dataset')]==dataIndc][('CODER','Percentage')]
    mean = np.round(np.mean(perc_ds)*100,2)
    std = np.round(np.std(perc_ds)*100,2)
    print(f"Dataset {dataIndc}---   Mean: {mean}%  Std: {std}%")


#%% md
# ## (Optional) Recoding by RA
#%% md
# ### Create file for RA recoding
#%% md
# It had come to our understanding that the label of "EVENT_accident" meant different things between coders. After a session of getting the coders onto the right page, we had one of our strongest coders LR, go through and recheck all EVENT codings by the other RAs. They coded 5,500 passages out of the ~10,000
# <br> If you are reading this on Github, the pathing of the raw RA files may be different.
#%%
RA = "LR"
RA_recode_dir = '../../../MEM-DEV-Stuff-for-RAs/Coding/'+RA+'/Recoding_EVENTS/'
df_recode_raw = pd.read_excel(RA_recode_dir+"Combined_Cultures.xlsx", sheet_name='Recoding Sheet', header=[0,1,2], engine='openpyxl')
df_recode_raw = df_recode_raw.drop(columns=df_recode_raw.columns[0])
df_recode_raw.head(2)
#%%
# Get only desired columns
df_recode = df_recode_raw.copy()
cols = ['No_Info', 'Illness', 'Accident', 'Other']
ra_recode_cols = [("Original RA Coding", 'CULTURE', 'Passage Number'), ("Original RA Coding", 'CULTURE', 'Passage')] + [("Original RA Coding", 'EVENT', col) for col in cols]
ra_recode_cols += [("Recoding", 'EVENT', col) for col in cols]
df_recode = df_recode[ra_recode_cols].dropna()

len(df_recode)
#%%
df_RA_Orig = df_recode["Original RA Coding"]["EVENT"][['No_Info', 'Illness', 'Accident', 'Other']]
df_RA_Recode = df_recode["Recoding"]["EVENT"][['No_Info', 'Illness', 'Accident', 'Other']]
assert len(df_RA_Orig) == len(df_RA_Recode), "Lengths do not match"
print("lengths of the two codings:", len(df_RA_Orig), len(df_RA_Recode))
#%%
from sklearn.metrics import cohen_kappa_score
columns = df_RA_Orig.columns
kappas_list= []
mainCol = ""
for col in columns:
    kappas = np.round(cohen_kappa_score(df_RA_Orig[col], df_RA_Recode[col]),4)
    if mainCol != col[0]:
        displayCol = col[0]
        mainCol = col[0]
    else:
        displayCol = len(mainCol)*' '
    kappas_list.append(kappas)
    print(f"{col}{' ' *(25-len(col))} {kappas}")
print("\nMean Kappas:", round(np.mean(kappas_list),4))