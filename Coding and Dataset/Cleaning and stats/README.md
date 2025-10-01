The folowing "cleaning" notebooks are used to clean the dataset from webscraping all the way to NLP analysis. For more in depth explanation of what each one does, consider checking out "End-to-End cleaning.docx" guide.

Here below is the order of events one should do in order from webscraping, to NLP transformer models (Lexical search requires its own order, see End-to-End cleaning.docx)
1. `eHRAF_Scraper` Scrape Yale's database, see https://github.com/Chantland/eHRAF_Scraper. Gives file "_Altogether_Dataset.docx"
2. `cleaning-eHRAF_Scraper.ipynb` clean raw scraper data. Gives file "_Altogether_Dataset_CLEANED.docx"
3. `cleaning-CultureSplitter.ipynb` split into multiple Culture datafiles for easier human coding as well as optionally getting a subset of the datasets based on the predicted distribution of desired labels via OCMs
4. `Human Coding` Obtain human input
5. `cleaning-RACode.ipynb` Clean human input. Gives file "_Altogether_Dataset_RACoded.docx"
6. `cleaning-DatasetConcat.ipynb` Combine multiple CLEANED human coding runthroughs ("_Altogether_Dataset_RACoded.docx") into a single cleaned dataset "_Altogether_Dataset_RACoded_Combined.docx"
7. `NLP` Use cleaned codings ("_Altogether_Dataset_RACoded_Combined.docx") to train and predict.



