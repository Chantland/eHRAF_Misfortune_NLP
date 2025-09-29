#%%
# from datasets.dataset_dict import DatasetDict
from datasets import Dataset, concatenate_datasets
# import evaluate

import pandas as pd
import numpy as np
from datetime import datetime
# from sklearn.model_selection import train_test_split 

#%%
# from huggingface_hub import notebook_login
# # If below code does not work, copy and paste this code in the terminal: huggingface-cli login 
# then paste your token


# notebook_login()
#%%
# load a list of passages and predict them (will take about .25 seconds per passage for me so beware the wait)
def predictor(data, labels, tokenizer_kwargs, classifier):
    dataOutput = []
    for text in data:
        # get actual labels
        actual_labels = [text[label] for label in labels]


        # get predicted labels
        prediction = classifier(text['passage'], **tokenizer_kwargs)
        scores = {item['label']:item['score'] for item in prediction[0]} #turn prediction into a dictionary
        pred_labels = [1 if scores[label] >= 0.5 else 0 for label in labels]

        
        output_dict = dict()
        output_dict["pred_labels"] = pred_labels
        output_dict["actual_labels"] = actual_labels
        output_dict["passage"] = text['passage']
        output_dict["ID"] = text['ID']


        # score[0][("actual_label", 'passage')] = text['passage'], text['label']
        dataOutput.append(output_dict)
    return dataOutput

# Get F1 scores
def score(dataOutput, labels):
    from sklearn.metrics import f1_score, accuracy_score

    df_score = pd.DataFrame(index=['NLP'], columns= [label+"_F1" for label in labels] + ["Micro_F1", "Macro_F1"])
    actual_labels = [x['actual_labels'] for x in dataOutput]
    pred_labels = [x['pred_labels'] for x in dataOutput]
    for index, label in enumerate(labels):
        f1 = round(f1_score(y_true=np.array(actual_labels)[:,index], y_pred=np.array(pred_labels)[:,index]),3)
        df_score.at['NLP', label+"_F1"] = f1
        # print(f"{label}: {(6 - len(label)) *' '}{f1}")

    # print("\n")

    f1_micro = round(f1_score(y_true=actual_labels, y_pred=pred_labels, average='micro'),3)
    f1_macro = round(f1_score(y_true=actual_labels, y_pred=pred_labels, average='macro'),3)
    df_score.at['NLP', "Micro_F1"] = f1_micro
    df_score.at['NLP', "Macro_F1"] = f1_macro
    return df_score
def cor_score(dataOutput, labels):
    from sklearn.metrics import matthews_corrcoef
    df_score = pd.DataFrame(index=['NLP'], columns= [label+"_Cor" for label in labels])
    actual_labels = [x['actual_labels'] for x in dataOutput]
    pred_labels = [x['pred_labels'] for x in dataOutput]
    for index, label in enumerate(labels):
        corrcoef = round(matthews_corrcoef(y_true=np.array(actual_labels)[:,index], y_pred=np.array(pred_labels)[:,index]),3)
        df_score.at['NLP', label+"_Cor"] = corrcoef    
    return df_score
    # print(f'F1 score (micro) {f1_micro}\nF1 score (macro) {f1_macro}')
#%% md
# ## Inference
#%% md
# ### Load datset 
#%%
import json
loc = ""
# loc = "../HRAF_MultiLabel_ThreeLargeClasses/" #load old threemain class (comment this out unless you specifically are using it)

# dataset = load_dataset('csv', data_files={'train': 'train.txt', 'validation': 'val.txt', 'test': 'test.txt'}, sep=";", 
#                               names=["text", "label"])


f = open(loc+"Datasets/test_dataset.json")
# f = open("../HRAF_MultiLabel_ThreeLargeClasses/Datasets/test_dataset.json") #load old threemain class (comment this out unless you specifically are using it)
data = json.load(f)
f.close()
Hraf = Dataset.from_dict(data)
Hraf
#%% md
# ### Define Kwargs and Labels
#%%
# Define tokenizer kwargs
tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}

classifier_kwargs = {'top_k':None, 'device':0} #Set device -1 for CPU, 0 or higher for GPU

# get label names
labels = [label for label in Hraf.features.keys() if label not in ['ID', 'passage']]
labels
#%% md
# ## Single Model Inference
#%% md
# Run this or the other model, not both
#%%
## Initial single run model, mostly derived from the simplistic multi-label classification task. You may delete but be cautious

import torch
import torch.nn as nn
from transformers import TrainingArguments, Trainer, DistilBertForSequenceClassification, PreTrainedModel, PretrainedConfig, DataCollatorWithPadding, DistilBertModel, AutoConfig


class SubLabelClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SubLabelClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)    ## Linear is a simple linear transformation layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)  
## One big mistake with NNs is making sure the input/output layers’ content/dimensions match up; 
# note the above has 2 Linear layers like this: input -> hidden; hidden -> output
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Create sample model
class HierarchicalMultiLabelClassifier(PreTrainedModel):
    def __init__(self, model_name, num_main_labels, num_event_labels, num_cause_labels, num_action_labels, config):
        super(HierarchicalMultiLabelClassifier, self).__init__(config)
        # # Not sure the purpose of these here, maybe a fluke with GPT or is for later reference donw the line, probably not needed here
        # self.num_main_labels = num_main_labels
        # self.num_event_labels = num_event_labels
        # self.num_cause_labels = num_cause_labels
        # self.num_action_labels = num_action_labels
        
        # self.distilbert = DistilBertForSequenceClassification.from_pretrained(model_name) 
        self.distilbert = DistilBertModel.from_pretrained(model_name, config=config) # This does not use the head for sequence classification, I am wondering if that is a disadvantage
        hidden_size = self.distilbert.config.hidden_size
        
        # logits for each classifier group (might only need main and sublabel groups...)
        self.main_classifier = nn.Linear(hidden_size, num_main_labels)
        self.event_classifier = SubLabelClassifier(input_dim = self.distilbert.config.hidden_size, hidden_dim = 50, output_dim = num_event_labels)
        self.cause_classifier = SubLabelClassifier(input_dim = self.distilbert.config.hidden_size, hidden_dim = 50, output_dim = num_cause_labels)
        self.action_classifier = SubLabelClassifier(input_dim = self.distilbert.config.hidden_size, hidden_dim = 50, output_dim = num_action_labels)
        #
    def forward(self, input_ids, attention_mask): #(should the argument params be "tensor" instead like the huggingface example?)
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]  # Last hidden state
        pooled_output = hidden_state[:, 0]  # [CLS] token
        
        main_logits = self.main_classifier(pooled_output)
        event_logits = self.event_classifier(pooled_output) #previously outputs.last_hidden_state as per chat gpt. It was changed also because of chat GPT but also because the sizes of the logits and the labels did not match 
        cause_logits = self.cause_classifier(pooled_output)
        action_logits = self.action_classifier(pooled_output)

        # #Delete and uncomment above, this is just to circumvent the head sequence classification issues
        # main_logits = None
        # event_logits = None
        # cause_logits = None
        # action_logits = None
        
        return main_logits, event_logits, cause_logits, action_logits
#%%
## Attempt to create a inference model which uses a hierarchical structure of classification. 
## The output structure from this looks somewhat good as far as I can tell, but the predictions are terrible leading me to believe it might not be working as intended
## You may delete but it is best to keep until we get a more working inference model
from transformers import Pipeline

class CustomClassificationPipeline(Pipeline):
    def __init__(self, model, tokenizer,  **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
    
    def __call__(self, texts, threshold=True, **kwargs):
        # Tokenize the input
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Get model predictions
        outputs = self.model(**inputs)

        # Apply sigmoid (since it's a multi-label classification problem)
        sigmoid = torch.nn.Sigmoid()
        main_logits, event_logits, cause_logits, action_logits = outputs

        # Convert logits to probabilities
        main_probs = sigmoid(main_logits).detach().cpu().numpy()
        event_probs = sigmoid(event_logits).detach().cpu().numpy()
        cause_probs = sigmoid(cause_logits).detach().cpu().numpy()
        action_probs = sigmoid(action_logits).detach().cpu().numpy()

        # Define label categories for each set of outputs
        main_labels = ["EVENT", "CAUSE", "ACTION"]
        event_labels = ["EVENT_Illness", "EVENT_Accident", "EVENT_Other"]
        cause_labels = ["CAUSE_Just_Happens", "CAUSE_Material_Physical", "CAUSE_Spirits_Gods", "CAUSE_Witchcraft_Sorcery", "CAUSE_Rule_Violation_Taboo","CAUSE_Other"]
        action_labels = ["ACTION_Physical_Material", "ACTION_Technical_Specialist", "ACTION_Divination", "ACTION_Shaman_Medium_Healer","ACTION_Priest_High_Religion","ACTION_Other"]

        # Generate the predictions for each category (if threshold is desired)
        if threshold:
            main_predictions = [(main_labels[i], prob) for i, prob in enumerate(main_probs[0]) if prob > 0.5]
            event_predictions = [(event_labels[i], prob) for i, prob in enumerate(event_probs[0]) if prob > 0.5]
            cause_predictions = [(cause_labels[i], prob) for i, prob in enumerate(cause_probs[0]) if prob > 0.5]
            action_predictions = [(action_labels[i], prob) for i, prob in enumerate(action_probs[0]) if prob > 0.5]
        elif threshold==False:
            main_predictions = [(main_labels[i], prob) for i, prob in enumerate(main_probs[0])]
            event_predictions = [(event_labels[i], prob) for i, prob in enumerate(event_probs[0])]
            cause_predictions = [(cause_labels[i], prob) for i, prob in enumerate(cause_probs[0])]
            action_predictions = [(action_labels[i], prob) for i, prob in enumerate(action_probs[0])]

        return {
            "main_predictions": main_predictions,
            "event_predictions": event_predictions,
            "cause_predictions": cause_predictions,
            "action_predictions": action_predictions
        }
    def _sanitize_parameters(self, **kwargs):
        """
        Handle parameters passed to the pipeline and return cleaned or adjusted versions.
        """
        preprocess_kwargs = {}
        forward_kwargs = {}
        postprocess_kwargs = {}
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs):
        """
        Preprocess the inputs (tokenization, truncation, etc.).
        """
        # Tokenize the input text
        return self.tokenizer(inputs, return_tensors="pt", truncation=True, padding=True, max_length=512)

    def _forward(self, model_inputs):
        """
        Forward pass through the model.
        """
        # Forward pass through the model
        outputs = self.model(**model_inputs)
        return outputs
    def postprocess(self, model_outputs):
        """
        Process the raw model outputs (logits) into human-readable format (e.g., probabilities, labels).
        """
        sigmoid = torch.nn.Sigmoid()

        # Assuming your model outputs a tuple of logits for different label categories
        main_logits, event_logits, cause_logits, action_logits = model_outputs

        # Convert logits to probabilities
        main_probs = sigmoid(main_logits).detach().cpu().numpy()
        event_probs = sigmoid(event_logits).detach().cpu().numpy()
        cause_probs = sigmoid(cause_logits).detach().cpu().numpy()
        action_probs = sigmoid(action_logits).detach().cpu().numpy()

        # Define label categories for each set of outputs
        main_labels = ["EVENT", "CAUSE", "ACTION"]
        event_labels = ["EVENT_Illness", "EVENT_Accident", "EVENT_Other"]
        cause_labels = ["CAUSE_Just_Happens", "CAUSE_Material_Physical", "CAUSE_Spirits_Gods", "CAUSE_Witchcraft_Sorcery", "CAUSE_Rule_Violation_Taboo","CAUSE_Other"]
        action_labels = ["ACTION_Physical_Material", "ACTION_Technical_Specialist", "ACTION_Divination", "ACTION_Shaman_Medium_Healer","ACTION_Priest_High_Religion","ACTION_Other"]

        # Generate the predictions for each category
        main_predictions = [(main_labels[i], prob) for i, prob in enumerate(main_probs[0]) if prob > 0.5]
        event_predictions = [(event_labels[i], prob) for i, prob in enumerate(event_probs[0]) if prob > 0.5]
        cause_predictions = [(cause_labels[i], prob) for i, prob in enumerate(cause_probs[0]) if prob > 0.5]
        action_predictions = [(action_labels[i], prob) for i, prob in enumerate(action_probs[0]) if prob > 0.5]

        # Return the predictions in a structured format
        return {
            "main_predictions": main_predictions,
            "event_predictions": event_predictions,
            "cause_predictions": cause_predictions,
            "action_predictions": action_predictions
        }
            
#%%
## DUMMY DELETE (try using custom pretrained model)
from transformers import AutoTokenizer, pipeline
model_path = "Model_1_BaseTest/Hierarchy_test_fold_1"
checkpoint_path = "checkpoint-10790"
import os
path =os.path.abspath(f"{model_path}/{checkpoint_path}")
model_name = path
num_main_labels = 3 # For EVENT, CAUSE, ACTION
num_event_labels = 3
num_cause_labels = 6
num_action_labels = 6
config = AutoConfig.from_pretrained(model_name)
model = HierarchicalMultiLabelClassifier(model_name, num_main_labels, num_event_labels, num_cause_labels, num_action_labels, config)
tokenizer = AutoTokenizer.from_pretrained(path)
# model = HierarchicalMultiLabelClassifier.from_pretrained(path)
#%%
## Use hierarchical version of inference and display a demo of the predictions
## dummy delete
classifier = CustomClassificationPipeline(model=model, tokenizer=tokenizer)
text = '''
“Drinking-tubes made of the leg-bones of swans (Fig. 109) are 190 also used chiefly as a measure of precaution against diseases subject to shunning.....”
'''
tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}

classifier_kwargs = {'top_k':None, 'device':0} #Set device -1 for CPU, 0 or higher for GPU
prediction = classifier(text, **tokenizer_kwargs)
prediction


#%%
# Attempt to use simplistic initial classification (note that the output is a single list which is likely not preferable.)
from transformers import pipeline, AutoTokenizer

# CHANGE Model name
# model = "Model_1_BaseTest/Hierarchy_test_fold_1"
# checkpoint_path = "checkpoint-10790"

model = "Model_1_BaseTest/Hierarchy_test_fold_1"
checkpoint_path = "checkpoint-10790"
# set up the pipeline from local
import os
path =os.path.abspath(f"{model}/{checkpoint_path}")
classifier = pipeline("text-classification", model=path, **classifier_kwargs)


# sample inference ENTER TEXT IN HERE.
text = '''
“Drinking-tubes made of the leg-bones of swans (Fig. 109) are 190 also used chiefly as a measure of precaution against diseases subject to shunning.....”
'''
# reveal sample classification
prediction = classifier(text, **tokenizer_kwargs)
prediction

# # Demo other models (COMMENT THIS OUT UNLESS YOU REALLY WANT TO DEMO THIS)
# # Set up path from online hub (note, this is analogous but different model and is here because this is a demo)
# classifier = pipeline("text-classification", top_k=None, model="Chantland/Hraf_MultiLabel", use_auth_token="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", tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"))
# model = "MultiLabel_ThreeLargeClasses"
#%% md
# ### Predict The Dataset
#%%
# Predict dataset (may take about .25 seconds per passage when tested on lab mac, could differ depending on your system)
# Also note that this pipeline is sequential and may give a warning saying it is unoptimized. Currently, using a whole dataset does not seem to reap faster results so we are remaining with sequential
HrafOutput = predictor(Hraf, labels=labels, tokenizer_kwargs=tokenizer_kwargs, classifier=classifier)
print(len(HrafOutput), "passages Predicted")
#%%
# HrafOutput = classifier(Hraf['passage'],**tokenizer_kwargs)
# print(len(HrafOutput), "passages Predicted")
#%% md
# ### Calculate "Correctness" Metrics
#%%
#get F1 scores for labels.
df_score = score(HrafOutput, labels)
# Optional TEST, get correlational score instead)
df_score = score(HrafOutput, labels)
df_score
#%%
# Optional TEST, get correlational score instead)
df_score_cor = cor_score(HrafOutput, labels)
df_score_cor
#%% md
# #### Add correctness to file
#%%

# export F1 scores to excel
df_scoresSep = df_score.copy()
# first load train (and maybe add validation)
f = open(loc+"Datasets/train_dataset.json")
data = json.load(f)
train = Dataset.from_dict(data)
if os.path.isfile(loc+"Datasets/validation_dataset.json"):
    f = open(loc+"Datasets/validation_dataset.json")
    data = json.load(f)
    valid = Dataset.from_dict(data)
    train = concatenate_datasets([train, valid])
# add lengths of test and training set
df_scoresSep[["test_length", "train_length"]] = (len(Hraf), len(train))
# add date
df_scoresSep.insert(0, "Date", [datetime.today().date()])
if loc == "":
    df_scoresSep['Notes'] = f"model: {model}/{checkpoint_path}, Dataset: {model}"
else:
    df_scoresSep['Notes'] = f"model: {model}/{checkpoint_path}, Dataset: {loc}"
# load model_performance.xlsx or else create it
if os.path.isfile("Model_Prediction_Performance.xlsx"):
    df_oldScores = pd.read_excel("Model_Prediction_Performance.xlsx", index_col=0)
    df_oldScores_merged = pd.concat([df_scoresSep, df_oldScores])
    nonDateCols = df_oldScores_merged.columns[df_scoresSep.columns != 'Date']
    if any(df_oldScores_merged.duplicated(subset=nonDateCols)): # don't append the data unless it is new
        print("Duplicated scores found, skipping new addition")
        df_scoresSep = df_oldScores.copy()
    else:
        df_scoresSep = df_oldScores_merged.copy()
        df_scoresSep['Date'] = df_scoresSep['Date'].astype('datetime64[ns]')
        df_scoresSep.to_excel("Model_Prediction_Performance.xlsx")
else:
    df_scoresSep['Date'] = df_scoresSep['Date'].astype('datetime64[ns]')
    df_scoresSep.to_excel(f"Model_Prediction_Performance.xlsx")
df_scoresSep
#%% md
# 
#%% md
# ## Checkpoint Multi-model Inference
#%% md
# This is to run over MANY models and checkpoints to test and see which is the strongest. This is ran instead of the single model one above and should NOT be ran together with the single model (simply because they do different things)
#%%
# code for running through all checkpoints
# code for running through all checkpoints
import os
import pandas as pd
import re
import json
from transformers import pipeline, AutoTokenizer
def checkpointInfer(path, data, labels, tokenizer_kwargs, classifier_kwargs, folds=True, output_str="output_dir_", modelDestinctifier:str= "ModelDistinctifierUnknown"):
    # Initiate Dataframe overall
    df = pd.DataFrame([])

    # Get all viable models
    # Makes sure the model starts with the output string and is a directory
    models = [name for name in os.listdir(path) if (name.startswith(output_str) and os.path.isdir(f"{path}/{name}"))]

    for model in models:
        # Initiate Dataframe for each model
        df_model = pd.DataFrame([])

        # Get checkpoint directory for a particular model and get the unit in which the model is distinguished (like learning rates)
        checkpoints_dir = [checkpoint for checkpoint in os.listdir(f"{path}/{model}") if checkpoint.startswith("checkpoint")]
        modelDestinctifier_unit = re.findall(f"{output_str}(.*?)_",model)
        try:
            modelDestinctifier_unit = float(modelDestinctifier_unit[0])
        except:
            pass


        # Predict for each checkpoint (within said model) and save results
        for checkpoint in checkpoints_dir:
            # Initiate Dataframe for each checkpoint
            df_checkpoint = pd.DataFrame([])
            # set up the pipeline from local
            model_path =os.path.abspath(f"{path}/{model}/{checkpoint}")
            classifier = pipeline("text-classification", model=model_path, **classifier_kwargs)
            # Get Predictions
            dataOutput = predictor(data, labels=labels, tokenizer_kwargs=tokenizer_kwargs, classifier=classifier)
            # Get scores
            df_checkpoint = score(dataOutput, labels)
            df_checkpoint = df_checkpoint.reset_index(drop=True) #remove the index here


            df_checkpoint.insert(0,modelDestinctifier,modelDestinctifier_unit) #insert model distinctifier (like weight decay or learning rate)
            #Extract and add Fold name if relevant
            if folds: #if using folds
                fold = re.findall(r"fold_(\d*)",model)
                fold = int(fold[0])
                df_checkpoint.insert(1,"Fold",fold)
            else:
                fold = ""

            # get checkpoint
            checkpoint_num = re.findall(r"checkpoint-(\d*)",checkpoint)
            assert len(checkpoint_num) == 1, f"More or less than one checkpoint numbers found: {len(checkpoint_num)} checkpoints"
            checkpoint_num = int(checkpoint_num[0])

            df_checkpoint.insert(0,"Model",model) # Add model name
            df_checkpoint.insert(0,"Checkpoint",checkpoint_num)
            df_model = pd.concat([df_model,df_checkpoint])
            print(model, checkpoint, "Complete")

        # concat model to overarching dataframe
        df = pd.concat([df,df_model])
        # save df for each model (as a checkpoint)
        # import evaluation if it exists
        if os.path.exists(f"{path}/Inference_Test.xlsx"):
            old_df = pd.read_excel(f"{path}/Inference_Test.xlsx", sheet_name="Sheet1", index_col=0)
            df_model = pd.concat([old_df, df_model])

        df_model.to_excel(f"{path}/Inference_Test.xlsx", sheet_name="Sheet1")
        print(model, "Successfully Saved")

    return df


            




# output_str="output_dir_"

# model = "MultiLabel_ThreeLargeClasses_kfoldsDEMO_WeightInvestigation"
# path =os.path.abspath(f"HRAF_Model_{model}")
# x = [name for name in os.listdir(path) if (name.startswith("output_dir_") and os.path.isdir(f"{path}/{name}"))]
# # x
# modelDestinctifier_unit = re.findall(f"{output_str}(.*?)_",x[1])
# try:
#     modelDestinctifier_unit = float(modelDestinctifier_unit)
# except:
#     pass
#%%
#This code will take a LONG time depending on how many models you have. It is recommended to use a GPU
path = loc+"/Model_3_LearningRates"
output_str = "Learning_Rate_"
modelDestinctifier = "Learning_Rate"

df_allScores = checkpointInfer(path=path, data=Hraf, labels=labels, tokenizer_kwargs=tokenizer_kwargs,  classifier_kwargs=classifier_kwargs, folds=True, output_str=output_str, modelDestinctifier= modelDestinctifier)
df_allScores
#%% md
# ## Optional File save
#%%
# HrafOutput
#%%
# optionally save the file to json
from transformers import AutoTokenizer
import copy

HrafOutput_dummy = copy.deepcopy(HrafOutput)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["passage"], truncation=True)

tokenized_Hraf = Hraf.map(preprocess_function, batched=True)

for index, passage in enumerate(HrafOutput_dummy):
    assert passage['passage'] == tokenized_Hraf[index]['passage']
    passage['pred_labels'] = {key:passage['pred_labels'][index] for index, key in enumerate(labels)}
    passage['actual_labels'] = {key:passage['actual_labels'][index] for index, key in enumerate(labels)}
    passage['input_ids'] = tokenized_Hraf[index]['input_ids']
#%%
import json
# Save to unformatted json (uncomment)
with open(f"Datasets/tokenized_inputs.json", "w") as outfile:
    json.dump(HrafOutput_dummy, outfile)


# # Save to Dataset (uncomment)
# HrafOutput_dummy_dataset = Dataset.from_list(HrafOutput_dummy)
# Dataset.to_json(HrafOutput_dummy_dataset, f"Datasets/tokenized_Hraf")
#%% md
# 
#%% md
# ## CHi Square
#%%
from scipy.stats import chi2_contingency

ct_EVENT_CAUSE = pd.crosstab(df[('EVENT','No_Info')], df[('CAUSE','No_Info')], rownames=['ACTION'], colnames=['CAUSE'])
ct_EVENT_CAUSE
#%%
def chi_square_calc(row, col):
    cross_tab = pd.crosstab(df[(row,'No_Info')], df[(col,'No_Info')], rownames=[row], colnames=[col])
    stat, p, dof, expected = chi2_contingency(cross_tab)
    results = f"{row} by {col}:\nchi: {round(stat,1)}\np:   {round(p,3)}\n\n"
    return results

group_list = [('EVENT', 'CAUSE'), ('EVENT', 'ACTION'), ('ACTION', 'CAUSE')]
for row, col in group_list:
    print(chi_square_calc(row, col))
#%%
def chi_sqr(obs):
    size_x = obs.shape
    chi_mat = np.zeros(size_x)
    for row in range(size_x[0]):
        for col in range(size_x[1]):
            exp = np.sum(x[row]) * np.sum(x[:,col]) / np.sum(x)
            chi_mat[row, col] = np.sum((obs[row, col] - exp)**2 / exp)
    return chi_mat

print(np.sum(chi_sqr(x)))
