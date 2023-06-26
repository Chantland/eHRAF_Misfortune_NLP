# eHRAF_Misfortune_NLP
 Natural language processing to mimic the coding of research assistants for misfortune related coding. For a quick demo of the model see  the online repository for the model <a href="https://huggingface.co/Chantland/HRAF_Event_Demo"> here! </a>

- For look into how the model was trained, use HRAF_Training.ipynb
- For basic inference see HRAF_Inference.ipynb.

## Current Rendition 6/01/2023
A Natural language model was run using 1750 passages prelabled by research assistants using the "No_Info" column for events which depicts if an event is present or not. 1460 passages from Culture_Coding_old.xlsx were used to train and 365 passages were used to test. Each was coded by a research assistant and given a label. 33 Passages were truncated to 512 tokens so there may be some inaccuracies in the model's training based on labels. Nonetheless, the same 1750 passages were refed into the model for inference and received an F1 score of .97. F1 scores take the harmonic mean of precision (which is sensitive to false positives) and recall (which is sensitive to false negatives). A score of .5 or higher is considered “ok” and a score of .9 or higher is considered “very good”. Our F1 score was extremely good but this is to be expected as these passages were the ones being trained on in the first place. Nonetheless, feeding in 140 completely new trained passaged gave an F1 score of .94, still very good!

