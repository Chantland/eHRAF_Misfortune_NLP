# eHRAF_Misfortune_NLP
 Natural language processing to mimic the coding of research assistants for misfortune related coding. For a quick demo of the model see  the online repository for the model <a href="https://huggingface.co/Chantland/HRAF_Event_Demo"> here! </a>

- For look into how the model was trained, use HRAF_Training.ipynb
- For basic inference see HRAF_Inference.ipynb.

## Current Rendition 6/01/2023
First steps of natural language processing. The EVENT column which depicts if an event is present or not was fed into the model to create a working text-classification model. 1460 passages from  `Culture_Coding_old.xlsx` were used to train and 365 were used to test. Each was coded by a research assistant and given a label. 33 Passages were truncated to 512 tokens so there may be some inaccuracies in the model's training based on labels. Nontheless, the same 1750 passage were refid into the model for inference and recieved an F1 score of .97. This is of course high as these passages were the ones being trained on in the first place. Nonethless feeding in 140 completely new trained passaged gave an F1 score of .94, still very good!
