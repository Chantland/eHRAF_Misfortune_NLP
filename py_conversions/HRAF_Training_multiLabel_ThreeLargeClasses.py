#%%
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import evaluate
import os
import json
import pandas as pd

from sklearn.model_selection import train_test_split 

#%% md
#  <font color="red">NOTE</font> This code is outdated and a new model shall be created. Culture_Coding.xlsx files have been changed to _Altogether_Dataset_RACoded.xlsx
#%%
from huggingface_hub import notebook_login
# copy and paste this code in the terminal: huggingface-cli login 
# then paste your token


notebook_login()
#%% md
# ## Import the dataset
#%%
df = pd.read_excel("../../RA_Cleaning/Culture_Coding.xlsx", header=[0,1], index_col=0)
df.head(3)
#%%


# subdivide into just passage and outcome
df_small = pd.DataFrame()
df_small[["ID","passage","EVENT","CAUSE","ACTION"]] = df[[('CULTURE', "Passage Number"), ('CULTURE', "Passage"), ('EVENT', "No_Info"), ('CAUSE', "No_Info"), ('ACTION', "No_Info")]]
# Flip the lable of "no_info"
df_small[["EVENT","CAUSE","ACTION"]]  = df_small[["EVENT","CAUSE","ACTION"]].replace({0:1, 1:0})

# Remove certain passages which should not be in training or inference (these are duplicates that had to be manually found by a human)
values_to_remove = [3252, 33681, 6758, 10104]
df_small = df_small[~df_small['ID'].isin(values_to_remove)]
df_small


# create train and validation/test sets
train, test_val = train_test_split(df_small, test_size=0.3, random_state=10)
# do it again to get the test and validation sets (15% = 50% * 30%)
test, validation = train_test_split(test_val, test_size=0.5, random_state=10)




# Create an NLP friendly dataset
Hraf = DatasetDict(
    {'train':Dataset.from_dict(train.to_dict(orient= 'list')),
     'validation':Dataset.from_dict(validation.to_dict(orient= 'list')),
     'test':Dataset.from_dict(test.to_dict(orient= 'list'))})
Hraf

#%% md
# ### Save partitioned dataset 
#%%
def make_dir(path):
    import os
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
    # Create a new directory because it does not exist
        os.makedirs(path)

# make folder if it does not exist yet
path = os.getcwd() + '/Datasets'
make_dir(path)
# save to Json
for key in Hraf.keys():
    Hraf_dict = Hraf[key].to_dict()
    with open(f"{path}/{key}_dataset.json", "w") as outfile:
        json.dump(Hraf_dict, outfile)
#%% md
# Make sure the training set is as biased as our groups (we want to train on as or less biased data as the groups they come from) <br>
# We are shooting for .85, .68 and .68 for EVENT, CAUSE, and ACTION respectively
# 
#%%
# extract the total proportion
def totalProportion(df, col, present=1):
    value_counts = df[col].value_counts()
    percentage = round(value_counts[present]/len(df)*100,2)
    return percentage

# extracts percentages per datafaframe
def colProportion(Hraf, col):
    percentage_list = []
    for dataframe in Hraf.keys():
        percentage_list += [round(sum(Hraf[dataframe][col]) / (len(Hraf[dataframe]))*100,2)]
    return percentage_list



# print bias per label
dataframe_keys= Hraf.keys()
labels = [label for label in Hraf['train'].features.keys() if label not in ['ID', 'passage']]
header = "                                TOTAL"
for key in dataframe_keys:
    header += f"     {key}"
print(header)
print('_'*(len(header)+4))
for col in labels:
    totalPercentage =  totalProportion(df_small, col)
    percentage_list =  colProportion(Hraf, col)
    spacing = 10
    percentage_str = f"{totalPercentage}{' '* (spacing-len(str(totalPercentage)))}"
    for index, key in enumerate(dataframe_keys):
        percentage_str += f"{(len(key)-5)*' '}{percentage_list[index]}{' '* (spacing-len(str(percentage_list[index])))}"
    print(f"{col}:{' ' * (30- len(col))} {percentage_str}")
#%% md
# ## Preprocess
#%% md
# Create labels for training and preprocessing
#%%

labels = [label for label in Hraf['train'].features.keys() if label not in ['ID', 'passage']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
id2label
#%% md
# load a DistilBERT tokenizer to preprocess the text field: <br>
#%% md
# Create a preprocessing function to tokenize text and truncate sequences to be no longer than DistilBERT’s maximum input length:<br>
# Guidelines were followed from NielsRogge found <a href= "https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb"> here </a>
#%%
from transformers import AutoTokenizer
import numpy as np


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_data(examples):
  # take a batch of texts
  text = examples["passage"]
  # encode them
  encoding = tokenizer(text, max_length=512, truncation=True) #max length for BERT is 512
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill numpy array
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()

  return encoding
#%% md
# To apply the preprocessing function over the entire dataset, use 🤗 Datasets map function. You can speed up map by setting batched=True to process multiple elements of the dataset at once:
#%%
# Tokenize data, remove all columns and give new ones
tokenized_Hraf = Hraf.map(preprocess_data, batched=True, remove_columns=Hraf['train'].column_names)
#%%
tokenized_Hraf
#%%
example = tokenized_Hraf['train'][1]
print(example.keys())
print(tokenizer.decode(example['input_ids']))

#%%
print(example['labels'])
[id2label[idx] for idx, label in enumerate(example['labels']) if label == 1.0]
#%%
# Number of passages longer than 512 tokens (and therefore truncated)
sequence_i = []
for i, tx in enumerate(tokenized_Hraf['train']):
    if len(tx['input_ids']) >= 512:
        sequence_i.append(i)
print('Number Truncated: ', len(sequence_i))
print(f'Percentage Truncated: {round(len(sequence_i)/len(tokenized_Hraf["train"])*100,1)}%')
print(sequence_i)
#%%
print(len(Hraf['train'][50]['passage'].split()))
print(len(tokenized_Hraf['train'][50]['input_ids']))
#%% md
# Now create a batch of examples using <a href="https://huggingface.co/docs/transformers/v4.29.0/en/main_classes/data_collator#transformers.DataCollatorWithPadding"> DataCollatorWithPadding</a>. It’s more efficient to dynamically pad the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.
#%%
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#%% md
# Set tokenized passages to PyTorch Tensor
#%%
tokenized_Hraf.set_format("torch")
#%% md
# ## Evaluate
#%% md
# Obtain F1 score for evaluation
#%%
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
    
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result
#%% md
# 
# ### Train
# Before you start training your model, create a map of the expected ids to their labels with id2label and label2id:
#%%
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    problem_type='multi_label_classification',
    num_labels = len(labels), 
    id2label=id2label, 
    label2id=label2id
)
#%%
#forward pass test
outputs = model(input_ids=tokenized_Hraf['train']['input_ids'][0].unsqueeze(0), labels=tokenized_Hraf['train'][0]['labels'].unsqueeze(0))
outputs
#%%
training_args = TrainingArguments(
    output_dir="HRAF_Model_MultiLabel_ThreeLargeClasses2",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_Hraf["train"],
    eval_dataset=tokenized_Hraf["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
#%%
trainer.train()
#%%
trainer.evaluate()
#%% md
# ## Dummy Inference
#%%

text = "If one falls sick or dies, the natives at once conclude he must have been bewitched, or bitten, or hurt by the devil-- eringa ."

encoding = tokenizer(text, return_tensors="pt")
encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

outputs = trainer.model(**encoding)
#%% md
# The logits that come out of the model are of shape (batch_size, num_labels). As we are only forwarding a single sentence through the model, the `batch_size` equals 1. The logits is a tensor that contains the (unnormalized) scores for every individual label.
#%%

logits = outputs.logits
logits.shape
#%% md
# To turn them into actual predicted labels, we first apply a sigmoid function independently to every score, such that every score is turned into a number between 0 and 1, that can be interpreted as a "probability" for how certain the model is that a given class belongs to the input text.
# 
# Next, we use a threshold (typically, 0.5) to turn every probability into either a 1 (which means, we predict the label for the given example) or a 0 (which means, we don't predict the label for the given example).
#%%

# apply sigmoid + threshold
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(logits.squeeze().cpu())
predictions = np.zeros(probs.shape)
predictions[np.where(probs >= 0.5)] = 1
# turn predicted id's into actual label names
predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
print(predicted_labels)
#%%
