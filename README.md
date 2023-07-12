# eHRAF_Misfortune_NLP
 Natural language processing to mimic the coding of research assistants for misfortune related coding. See HRAF_NLP folder for all models.
- For look into how the model was trained, use HRAF_Training.ipynb (note the difference between singleLabel and multiLabel training sets)
- For basic inference see HRAF_Inference.ipynb. (note the difference between singleLabel and multiLabel training sets)
- For N-gram exploration consider checking out the N-gram Exploration folder. 

## Single label 6/01/2023
A Natural language model was run using 1750 passages prelabled by research assistants using the "No_Info" column for events which depicts if an event is present or not. 1400 passages from Culture_Coding_old.xlsx were used to train and 350 passages were used for validation. Each was coded by a research assistant and given a label. 110 training Passages were truncated to 512 tokens so there may be some inaccuracies in the model's training based on labels. Nonetheless, the same 1750 passages were refed into the model for inference and received an F1 score of .97. F1 scores take the harmonic mean of precision (which is sensitive to false positives) and recall (which is sensitive to false negatives). A score of .5 or higher is considered “ok” and a score of .9 or higher is considered “very good”. Our F1 score was extremely good but this is to be expected as these passages were the ones being trained on in the first place. Nonetheless, feeding in 140 completely new test passaged gave an F1 score of .94, still very good! For a quick demo of the model see  the online repository for the model <a href="https://huggingface.co/Chantland/HRAF_Event_Demo"> here! </a>



## Multiple Label 7/11/2023
A Natural Language model was run using the 1750 passages previously used for the single label run. These passages had labels for EVENT, CAUSE, and ACTION which indicate the presence of a misfortunate event, a cause for the misfortune, and an action relating to the remedy or reaction to a misfortunate event. Otherwise, everything is identical to the single run (see above for more details). Using a multiplable text classification, the model achieved an evaluation micro f1 score of .898 on the validation set. Running the model on a completely separate test set of 140 passages achieved a F1 score of .835 macro and .838 micro F1 score. Note that macro F1 scores are good for inbalanced classes while micro is better for balanced. Individual F1 scores per labels were 
<br>EVENT:  0.914
<br>CAUSE:  0.797
<br>ACTION: 0.794
<br> For a quick demo of the model see  the online repository for the model <a href="https://huggingface.co/Chantland/HRAF_MultiLabel"> here! </a>
