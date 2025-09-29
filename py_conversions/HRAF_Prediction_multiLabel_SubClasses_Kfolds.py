#%% md
# # Prediction
#%% md
# This uses most of the same methods found in "inference" jupyter notebook only that we only care what the prediction is and not what the "actual" answer is.<br>
# If you already have the file "_Node_NLP_predictions.xlsx", skip to the [Predicting](#Predicting) section
#%% md
# The following code chunks below should all be ran regardless of if you use the [Set Up](#set-up) or not
#%%
import pandas as pd
import numpy as np
import re

# for stripping OCMs (if relevant)
def OCM_stripper(df, OCM='OCM'):
    df[OCM] = df[OCM].apply(lambda x: re.sub(" |\'",'',x))
    df[OCM] = df[OCM].apply(lambda x: x[1:-1].split(','))
    return df
#%%
# CHANGE Folder where your files are located
folder = '(subjects-(contracts_OR_disabilities_OR_disasters_OR_friendships_OR_gift_giving_OR_infant_feeding_OR_lineages_OR_local_officials_OR_luck_and_chance_OR_magicians_and_diviners_OR_mortuary_specialists_OR_nuclear_family_OR_priesthood_OR_prophet'
directory = f"../../../eHRAF_Scraper-Analysis-and-Prep/Data/{folder}/"
#%% md
# Load transformer to get labels. If you do not need to do the [Set Up](#set-up) section and just want to see prediction slices, you can manually enter the labels in the [Predicting](#Predicting) section
#%%
from transformers import pipeline, AutoTokenizer

# set up the pipeline from Hugging Face (optional)
# f = open('../_HuggingFace_Auth.txt') #get the hugging face token, otherwise, enter your own in
# f = f.readlines()[0]
# classifier = pipeline("text-classification", top_k=None, model="Chantland/Hraf_MultiLabel", use_auth_token=f, tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"))


# Set up pipeline and classifier through existent folder
# Define tokenizer kwargs
tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}

classifier_kwargs = {'top_k':None, 'device':0} #Set device -1 for CPU, 0 or higher for GPU

# CHANGE model_name name
model_name = "model_5_Roberta/Learning_Rate_2e-05_Weight_Decay_0.01_fold_1"
checkpoint_path = "checkpoint-12450"
# set up the pipeline from local
import os
path =os.path.abspath(f"{model_name}/{checkpoint_path}")
classifier = pipeline("text-classification", model=path, **classifier_kwargs)


# sample inference ENTER TEXT IN HERE.
text = '''
“Drinking-tubes made of the leg-bones of swans (Fig. 109) are 190 also used chiefly as a measure of precaution against diseases ‘subject to shunning.’....”
'''
# reveal sample classification
prediction = classifier(text, **tokenizer_kwargs)
prediction


#%%

# Get labels
labels = [x['label'] for x in prediction[0]] 
labels
#%% md
# ## Predict
#%% md
# Functions and code used to do inference. Run all of these cells before using the functions.
#%% md
# This code will predict any dataset you give it so long as you have a "passage" and "ID" columns. Expect to take ~400 passages per minute
#%%


def dataframeCreator(df, labels, passage_name, ID_name:str=False, culture_name:str=False, OCM_name:str=False, values_to_remove:list=False):
    df_small = pd.DataFrame()
    # create columns
    assert isinstance(passage_name, str), "Need to supply the column name that you will be using to predict text as a string"
    # add ID column if present
    if ID_name is not False:
        assert isinstance(ID_name, str), "Need to supply the ID column header as a string"
        df_small["ID"] = df[ID_name]

        # use a list of integers to remove specific ID passages
        if values_to_remove is not False:
            df_small = df_small[~df_small['ID'].isin(values_to_remove)]
    # add culture column if present
    if culture_name is not False:
        assert isinstance(culture_name, str), "Need to supply the culture column header as a string"
        df_small["culture"] = df[culture_name]
    # add OCM column if present
    if OCM_name is not False:
        assert isinstance(OCM_name, str), "Need to supply the OCM column header as a string"
        df_small["OCM"] = df[OCM_name]
        # Turn the string of column OCM back into a list 
        df_small = OCM_stripper(df_small)

    df_small["passage"] = df[passage_name]

    # create columns based off of the labels we will use to predict the text
    df_small[labels] = np.nan
    
    return df_small


#%%
def predict(df, labels, passage_name:str="passage", tokenizer_kwargs=tokenizer_kwargs):

    passage_list = df[passage_name]

    
    for index, text in enumerate(passage_list):

        prediction = classifier(text, **tokenizer_kwargs)

        # get predictions
        scores = {item['label']:item['score'] for item in prediction[0]} #turn prediction into a dictionary
        pred_labels = [1 if scores[label] >= 0.5 else 0 for label in labels]
        df.loc[index, labels] = pred_labels
    return df
#%% md
# ### Initial Dataframe SetUp
#%% md
# Extract passages
#%%

# Old code for having a small subset of the columns, but I decided it may jsut be better to include all the fluff
# # Extract passages

# # CHANGE All of these depending on your dataframe locationand its column names
# df = pd.read_excel(directory+"_Altogether_Dataset_CLEANED.xlsx")
# passage_name = "Passage"
# ID_name = "Passage Number" #OPTIONAL but reccommended for MISF datasets
# culture_name = "Culture" #OPTIONAL but reccommended for MISF datasets
# OCM_name = "OCM" #OPTIONAL but reccommended for MISF datasets

# # fit up the dataframe so it can be more easily used for making predictions
# df_small = dataframeCreator(df=df, labels=labels, passage_name=passage_name, ID_name=ID_name, culture_name=culture_name, OCM_name=OCM_name)

# # # here is the basic version if you don't want ID's or cultures
# # df_small = dataframeCreator(df=df, labels=labels, passage_name=passage_name)

# df_small.head(4)


# Extract passages (include all the other column fluff)
df_small = pd.read_excel(directory+"_Altogether_Dataset_CLEANED_ALLSCRAPED.xlsx")
df_small[labels] = np.nan #preallocate space
df_small = OCM_stripper(df_small)  
df_small = df_small.rename(columns={"Passage Number":"ID"}) #Make Passage Number standardized to ID to match with machine learning labels (and more future proof, change Passage Number should this column name be discontinued or changed prior)

#Add in checkpoint Info Column to indicate what model is Predicting this

df_small["Model Info"] = None
df_small.loc[0, "Model Info"] = f"Model: {model_name}"
df_small.loc[1, "Model Info"] = f"Checkpoint: {checkpoint_path}"

df_small
#%% md
# Optionally add the dataset indicators if they exist (otherwise disregard this cell)
#%%
df_small.columns
#%%

df_dataset = pd.read_excel(directory+"_Dataset_Lists.xlsx")
df_dataset = df_dataset.rename(columns={'Passage Number':'ID'}) #rename
df_dataset = df_dataset[["ID","Dataset"]] # only use the following columns

df_small = df_small.merge(df_dataset, on='ID', how='left')
print(df_small["Dataset"].value_counts(dropna=False))
print(f"Total: {len(df_small)}")
#%% md
# ### Sample run
#%%
# Uncomment for sample run
df_shaved = df_small.iloc[0:200].copy()
df_shaved = predict(df_shaved, labels, passage_name="Passage", tokenizer_kwargs=tokenizer_kwargs)
df_shaved.head(4)
#%% md
# 
# ### Set Up coding for dataset
#%%
# Ranges from doing 60 to 400 passages a second but results may vary
df_coded = df_small.copy()
df_coded = predict(df_coded, labels, passage_name="Passage", tokenizer_kwargs=tokenizer_kwargs)
df_coded.to_excel(directory+"_Node_NLP_predictions.xlsx", index=False)
df_coded
#%% md
# ## Exploration (Optional)
# 
#%% md
# Each of these code chunks is for a different bisection of the data. The only required chunk is the one right below
#%%
# load dataset
df = pd.read_excel(directory+"_Node_NLP_predictions.xlsx")
if 'OCM' in df.columns:
    df = OCM_stripper(df)

def label_percentage(df,labels, grouping:str=False, OCM_grouping:list=False):
    df_perc = pd.DataFrame(columns=["grouping","count"])
    df_perc[labels] = np.nan

    # get total percentage per label
    df_perc.loc[0, "grouping"] = "TOTAL"
    df_perc.loc[0, "count"] = len(df)
    df_perc.loc[0, labels] = [df[label].mean() for label in labels]

    # get percentage by group per label
    if grouping is not False:
        assert isinstance(grouping, str), "Need to supply a string for the column group" # assert is used to quickly check if the code is working the way it is, if not, crash on purpose and give this string
        assert OCM_grouping is False, "Cannot do OCM grouping and normal grouping!"
        for index, group in enumerate(df[grouping].unique()):
            df_perc.loc[index+1, "grouping"] = group
            df_perc.loc[index+1, "count"] = len(df.loc[df[grouping]== group])
            df_perc.loc[index+1, labels] = [df.loc[df[grouping]== group][label].mean() for label in labels]
    # Grouping by OCMs. These are unique and cannot group them otherwise
    if OCM_grouping is not False: 
        assert isinstance(OCM_grouping, list), "OCMs must be in a list"
        print("Note, TOTAL will not add up as there is overlap between OCMs")
        for index, OCM in enumerate(OCM_grouping):
            df_perc.loc[index+1, "grouping"] = OCM
            msk = df['OCM'].apply(lambda x: not set(x).isdisjoint([OCM]))
            df_perc.loc[index+1, "count"] = len(df.loc[msk])
            df_perc.loc[index+1, labels] = [df.loc[msk][label].mean() for label in labels]
    return df_perc
#%% md
# ### Complete datafile prediction
#%%
df_perc = label_percentage(df,labels)
df_perc
#%% md
# ### By Culture Prediction
#%%
df_perc = label_percentage(df,labels, grouping="culture")
df_perc
#%% md
# ### By dataset
#%%
df_perc = label_percentage(df,labels, grouping="Dataset")
df_perc
#%% md
# ### Remove 788 OCM the check by Dataset
#%%
df_788removed = df.copy()

msk = (df_788removed['OCM'].apply(lambda x: set(x).isdisjoint(['788'])) | (df_788removed['Dataset'] == 1))
df_788removed = df_788removed.loc[msk].copy()
df_788removed
#%%
df_perc = label_percentage(df_788removed,labels, grouping="Dataset")
df_perc
#%% md
# ### Prediction by each OCM
# 
#%%
OCMs = ["750", "751", "752", "753", "780", "781", "784", "785", '586' , '684' , '688' , '731' , '732' , '756' , '767' , '777' , '791' , '792' , '793' , '431' , '572' , '594' , '613' , '624' , '675' , '853'] 
# OCMs = ["750", "751", "752", "753", "780", "781", "784", "785", "788"]
df_perc = label_percentage(df,labels, grouping=False, OCM_grouping=OCMs)
df_perc
#%%
