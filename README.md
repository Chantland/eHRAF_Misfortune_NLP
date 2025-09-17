# eHRAF_Misfortune_NLP
## Info
Cataloged here are transformer-based Natural Language processing Models that attempt to determine the prevalence of certain types of misfortunate EVENTS, the types of CAUSES for the misfortune, and the types of remedies or preventative ACTIONS. Importantly, there may be more than one misfortune occurring, as well as multiple causes and actions. We want to determine if certain misfortunes are explained as being the result of particular causes (like a stomach ache caused by spirits), and with this co-occurrence, do cultures recruit particular specialists to resolve or prevent these misfortunes?<br><br>
Our current model, described below in the history section <b>Multiple label Subclasses - Training and parameter testing 5/16/2024</b>, is a multi-label sequence classification model leveraging roBERTa transformer model to predict 12 labels.<br>
See section below for more info<br><br>

Additional info about this repository:
- Current definitive model "HRAF_Multilabel_SubClasses" found here <a href="https://github.com/Chantland/eHRAF_Misfortune_NLP/tree/main/HRAF_NLP/HRAF_MultiLabel_SubClasses_Kfolds"> HRAF_NLP/HRAF_MultiLabel_SubClasses_Kfolds </a>. This model predicts 
- For a look into how models were trained, see HRAF_Training.ipynb within each of the model folders. Model Inference testing for each model uses "HRAF_Inference.ipynb" files
- For a simple model using N-grams and Lexical search, see the Lexical Search and N-gram Exploration folders.
- Hierarchical model can be found in this repository, but it likely has issues, so it should not be used as a definitive version.
- For human-coded dataset, see "_Altogether_Dataset_RACoded_Combined.xlsx" found in <a href="https://github.com/Chantland/eHRAF_Misfortune_NLP/tree/main/Coding%20and%20Dataset/Coded%20Dataset"> Coding and Dataset</a> 


## History
### Single label 6/01/2023
A Natural language model was run using 1750 passages prelabeled by research assistants using the "No_Info" column for events, which depicts if an event is present or not. 1400 passages from Culture_Coding_old.xlsx were used to train and 350 passages were used for validation. Each was coded by a research assistant and given a label. 110 training Passages were truncated to 512 tokens, so there may be some inaccuracies in the model's training based on labels. Nonetheless, the same 1750 passages were refed into the model for inference and received an F1 score of .97. F1 scores take the harmonic mean of precision (which is sensitive to false positives) and recall (which is sensitive to false negatives). A score of .5 or higher is considered “ok” and a score of .9 or higher is considered “very good”. Our F1 score was extremely good, but this is to be expected as these passages were the ones being trained on in the first place. Nonetheless, feeding in 140 completely new test passages gave an F1 score of .94, still very good! 


### Multiple Label 7/11/2023
A Natural Language model was run using the 1750 passages previously used for the single label run. These passages had labels for EVENT, CAUSE, and ACTION which indicate the presence of a misfortunate event, a cause for the misfortune, and an action relating to the remedy or reaction to a misfortunate event. Otherwise, everything is identical to the single run (see above for more details). Using a multi-lable text classification, the model achieved an evaluation micro f1 score of .898 on the validation set. Running the model on a completely separate test set of 140 passages achieved an F1 score of .835 macro and .838 micro F1 score. Note that macro F1 scores are good for balanced classes while micro is better for unbalanced such as ours. Individual F1 scores per label were 
<br>EVENT:  0.914
<br>CAUSE:  0.797
<br>ACTION: 0.794
<br> For model see  the online repository for the model <a href="https://huggingface.co/Chantland/HRAF_MultiLabel"> here! </a>


### Multiple Label With Subclass 7/19/2023
This is a demo model with all of the main classes (“EVENT”, “CAUSE”, “ACTION” using the reverse coded column “No_Info”) and all the 15 subclasses. The model was trained and validated using 1746 passages (1396 for training, 350 for validation). This number is 4 off from the previous version as a Ngram decoder found 4 instances of passages which appeared to be subsets of other passages (e.g. “I played in the forest” is the subset of “John fetched some water. I played in the forest”). The model achieved a micro F1 score of .79 for its validation set which is “okay” approaching “good”. However, since the validation set is used to train the model, it is not an unbiased representation of success. Instead using a test set of 140 passages which were completely new and untested. We achieved a micro F1 score of .68 which is worse but decidedly still “okay”. However, this low score is likely due to the small testing set which does not give many chances to approximate towards the true f1 score (some subclasses don’t even have a single true positive) and is exasperated by the fact that many of the subclasses are extremely negatively biased. 
<br> For model see the online repository for the model <a href="https://huggingface.co/Chantland/HRAF_Multilabel_SubClasses"> here! </a>

### Multiple Label 12/01/2023
This is a demo using the new dataset which was less skewed towards containing misfortune. This model adds the addition to k-folds cross-validation to improve model performance but potentially may be overfitting. Regardless, only the main categories were used like in "Multiple Label" version. <br>
The current F1 micro score of
750 passages not used for training is .816. individual class f1 scores shown below.
<br>EVENT:  0.883
<br>CAUSE:  0.812
<br>ACTION: 0.733



### Multiple Label 1/13/2024
Due to our original dataset being heavily biased, we searched via our eHRAF scraper various subjects that are still interesting to us but are not directly biased towards misfortune. Additionally, we changed the run #1 (Sickness only dataset) by replacing “Jealousy Evil Eye” to “Other” in order to not only have the missing “other” category which was missed, but to also match with run #2 (Sickness plus Non-sickness  dataset) columns. Additionally, “evil eye” was not a common coding to begin with. 
We trained with 4340 passages via a kfold of 5.
F1 micro score of 1085 passages not used for training was .865.


### Multiple Label  Additional training 4/02/2024
Additional training added to model via adding more passages to present model rather than starting from scratch. We trained with 7277 passages via a kfold of 5.
F1 micro score of 1085 passages not used for training was .851.
<br>EVENT: 0.907
<br>CAUSE: 0.822
<br>ACTION: 0.805
<br> For model <a href="https://huggingface.co/Chantland/Hraf_Multilabel_K-foldsCrossValDemo"> here! </a>

### Multiple label Subclasses - Training and parameter testing 5/16/2024
We trained a series of multi-label classification models to detect 12 labels indicating the EVENT, CAUSE, and ACTION of misfortune (reduced from the original of 15), but removing the higher order labels (general presence of EVENT, CAUSE, and ACTION) as we wanted to detect specific labels. This was seen as our end goal, thus an accurate model would allow us to find co-occurance with specific themes of misfortune and how they were managed by individual cultures, thus giving greater understanding for how cultures ascribe misfortune; a task infeasible by hand given the large amount of data in anthropological datasets. 
<br>
We tuned hyperparameters like weight decay, dropout, learning rate, batch, as well as investigating the best transformer model to train on. As a result, we increased our F1 score from .2 to .66 while also improving per-label accuracy. We used a training set of 6634 human-coded passages, 1659 passages for the validation set, and 2074 for the test set. The most promising model transformed off roBERTa based and achieved the following F1 scores per label: 
  <li>EVENT:  -
    <ul>
      <li>
        Illness:  .876
      </li>
      <li>
        Accident:  .458
      </li>
      <li>
        Other:  .588
      </li>
    </ul>
  </li>
  <li>CAUSE:  -
    <ul>
      <li>
        Just Happens:  -
      </li>
      <li>
        Material Physical:  .476
      </li>
      <li>
        Spirits and Gods:  .728
      </li>
      <li>
        Witchcraft and Sorcery:  .651
      </li>
      <li>
        Rule Violation Taboo:  .517
      </li>
      <li>
        Jealous Evil Eye:  -
      </li>
    </ul>
  </li>
  <li>ACTION:  -
    <ul>
      <li>
        Physical Material:  .672
      </li>
      <li>
        Technical Specialist:  .5
      </li>
      <li>
        Divination:  .406
      </li>
      <li>
        Shaman Medium Healer:  .582
      </li>
      <li>
        Priest High Religion:  .375
      </li>
      <li>
        Other:  -
      </li>
    </ul>
  </li>
<br>
Please see model scripts within the Github folder HRAF_MultiLabel_SubClasses_Kfolds, model can be downloaded  <a href="https://huggingface.co/Chantland/HRAF_Multilabel_SubClasses"> here</a> or found here <a href="https://github.com/Chantland/eHRAF_Misfortune_NLP/tree/main/HRAF_NLP/HRAF_MultiLabel_SubClasses_Kfolds"> HRAF_NLP/HRAF_MultiLabel_SubClasses_Kfolds </a>
