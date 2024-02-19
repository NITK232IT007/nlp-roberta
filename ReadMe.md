# NLP Text summarization task with ROUGE Evaluation

This repository contains code for performing Natural Language Processing (NLP) tasks with evaluation using ROUGE (Recall-Oriented Understudy for Gisting Evaluation). Specifically, it includes an IPython Notebook file (`RoBERTa_Text_Summarization.ipynb`) that demonstrates how to perform NLP tasks such as text summarization or machine translation and evaluate the results using ROUGE metrics.

## Requirements

To run the IPython Notebook file (`RoBERTa_Text_Summarization.ipynb`), you need the following dependencies:

pip install rouge

install java (for evaluating ROUGE using package)

## Finetuned Model, code and dataset

Download the finetuned Model, code and dataset from drive: https://drive.google.com/drive/folders/1X2pWfIfqZhC_EVNM08GetbrhAJHWduzH?usp=sharing

## Initial code setup

Setting Up Google Drive
To use Google Drive for storing datasets and models, follow these steps:

Create a folder named NLP_Roberta_Final in your Google Drive ->  MyDrive .
Upload the fns2020 dataset into the NLP_Roberta_Final folder with name 'fns2020_dataset'.
Upload the provided RoBERTa model (Roberta_Classification_Model_Fine_Tuned) into the NLP_Roberta_Final folder. (To avoid excess training time).
You can then access these files from your notebook by mounting Google Drive and specifying the file paths accordingly.

## Evaluating the ROUGE Scores after generating summaries
For the evaluation we aim to use the Rouge Java package.

download the package from: 
https://github.com/kavgan/ROUGE-2.0

we are generating summaries according to the naming convention given by the contest authorities.
copy all the generated summaries from system1 folder -> put it in rouge2_v1.2.2_runnable/v1.2.2/projects/test-summarization/system and in that directory
run  command in v1.2.2: java -jar rouge2-1.2.2.jar.

This will generate results.csv in v1.2.2, do similarly for system2 summaries as well.

