#!/usr/bin/env python
# coding: utf-8


# from google.colab import drive
# drive.mount('/content/drive')


import os
import shutil
import os
# Pandas library for data manipulation and analysis
import pandas as pd
# Numpy library for numerical operations
import numpy as np
# Regular expression module for pattern matching and text processing
import re
# Natural Language Toolkit for text processing tasks
import nltk
# For tokenization
nltk.download('punkt')
# Tokenizer for word tokenization
from nltk.tokenize import word_tokenize
# Time module for time-related functions
import time
# Abstract Syntax Trees module for parsing Python code
import ast
# Train-test split function for splitting data
from sklearn.model_selection import train_test_split
# Transformers library for RoBERTa model
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
# DataLoader and Dataset classes for handling data
from torch.utils.data import DataLoader, Dataset
# tqdm library for progress bars
from tqdm import tqdm
# PyTorch library for deep learning
import torch
# Accuracy score metric for classification tasks
from sklearn.metrics import accuracy_score
# Automatic Mixed Precision for improved performance
from torch.cuda.amp import autocast, GradScaler
# Metrics for evaluating classification performance
from sklearn.metrics import confusion_matrix, classification_report
# ngrams function for generating n-grams from text
from nltk import ngrams
# Counter class for counting occurrences of elements
from collections import Counter
# Functional module for PyTorch operations
import torch.nn.functional as F
# Garbage collection module for memory management
import gc



def print_directory_tree(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        depth = dirpath.count(os.sep) - root_dir.count(os.sep)
        indent = ' ' * 4 * depth
        print(f"{indent}{os.path.basename(dirpath)}/")
        for filename in filenames:
            print(f"{indent}    {filename}")

def find_leaf_directories(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if the current directory is a leaf directory and contains text files
        if not dirnames and any(file.endswith(".txt") for file in filenames):
            yield dirpath

def create_summary_directories(root_dir):
    for leaf_directory in find_leaf_directories(root_dir):
        # Create "Summary" directory in the parent directory
        parent_dir = os.path.dirname(leaf_directory)
        summary_dir = os.path.join(''.join(parent_dir.split('/')[:-1]), "Summary")
        os.makedirs(summary_dir, exist_ok=True)

        # Copy text files from leaf directory to "Summary" directory with _summary.txt suffix
        for file in os.listdir(leaf_directory):
            if file.endswith(".txt"):
                original_file_path = os.path.join(leaf_directory, file)
                summary_file_name = os.path.splitext(file)[0] + "_summary.txt"
                summary_file_path = os.path.join(summary_dir, summary_file_name)

                # Copy with the new filename
                shutil.copy(original_file_path, summary_file_path)


# #### Common functions for Solution 1 and Solution 2 methods

# ### i) Solution 1 : Confidence score method

# In[2]:


import sys
sys.setrecursionlimit(10**6)




def preprocess_text(text):
    '''
    Function to preprocess the text, Remove extra spaces, Convert to lower case, and Remove special characters
    '''
    text = ' '.join(text.split())
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def preprocess_text_for_generatedSummary(text):
    '''
    Function to preprocess the text, Remove extra spaces, and Remove special characters for summary
    '''
    text = ' '.join(text.split())
    text = re.sub(r'[^a-zA-Z0-9.\s]', '', text)

    return text


def divide_into_segments(text, segment_length=250):
    '''
    Function to divide text into segments of 250 words
    '''
    words = word_tokenize(text)
    segment_length_words = max(1, min(segment_length, len(words)))
    segments = [words[i:i + segment_length_words] for i in range(0, len(words), segment_length_words)]
    return segments

def divide_into_segments_summary(text, segment_length=250):
    '''
    Function to divide text into segments of 250 words for summary
    '''

    words = word_tokenize(text)
    segment_length_words = max(1, min(segment_length, len(words)))
    segments = []
    current_segment = []
    for word in words:
        if word == '.':
            if current_segment and current_segment[-1] != '.':
                current_segment[-1] += '.'
        else:
            current_segment.append(word)
        if len(current_segment) >= segment_length_words:
            segments.append(current_segment)
            current_segment = []
    if current_segment:
        segments.append(current_segment)
    return segments

def count_bigrams(segment_text):
    # Tokenize the text into words
    words = nltk.word_tokenize(segment_text)

    # Generate bigrams
    bigrams = list(ngrams(words, 2))

    # Count the occurrences of each unique bigram
    bigram_counts = Counter(bigrams)

    # Use the length to get the count of unique bigrams
    unique_bigram_count = len(bigram_counts)

    return unique_bigram_count

def generate_summaries_roberta(test_data_path):
    file_paths = []
    summary_paths =[]
    texts = []
    seg_nums = []
    segment_summaries = []
    cnt = 0
    for leaf_directory in find_leaf_directories(test_data_path):
        # Create "Summary" directory in the parent directory
        parent_dir = os.path.dirname(leaf_directory)
        summary_dir = os.path.join(parent_dir, "Summary")
        os.makedirs(summary_dir, exist_ok=True)

        # Copy text files from leaf directory to "Summary" directory with _summary.txt suffix
        for file in os.listdir(leaf_directory):
            if file.endswith(".txt"):
                original_file_path = os.path.join(leaf_directory, file)
                summary_file_name = os.path.splitext(file)[0] + "_summary.txt"
                summary_file_path = os.path.join(summary_dir, summary_file_name)
                # Read each text file
                #file_path = os.path.join(root, file)
                #if file_path not in selected_files:
                    #continue
                with open(original_file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                preprocessed_text_summary = preprocess_text_for_generatedSummary(text)
                # Divide into segments
                segments_summary = divide_into_segments_summary(preprocessed_text_summary)
                # Append to lists
                segs_summary = [i+1 for i in range(len(segments_summary))]
                # Pre-process the text
                preprocessed_text = preprocess_text(text)
                # Divide into segments
                segments = divide_into_segments(preprocessed_text)
                # Append to lists
                segs = [i+1 for i in range(len(segments))]
                # Preprocess for candidate summary
                
                file_paths.extend([original_file_path] * len(segments))
                summary_paths.extend([summary_file_path]* len(segments))
                seg_nums.extend(segs)
                texts.extend(segments)
                segment_summaries.extend(segments_summary)



    # Create a DataFrame
    df = pd.DataFrame({'Report_File_Path': file_paths,'Summary_File_Path': summary_paths, 'Segment_No': seg_nums,'Segment_Text': texts, 'segments_for_summary':segment_summaries})
    #df['Segment_Text'] = df['Segment_Text'].apply(ast.literal_eval)
    df['Segment_Text'] = df['Segment_Text'].apply(lambda x: ' '.join(x))
    df['segments_for_summary'] = df['segments_for_summary'].apply(lambda x: ' '.join(x))
    # Display the DataFrame


    # df.to_csv('./Test_Annual_validation_data_segments'+'.csv', index=False)
    # df = pd.read_csv('./Test_Annual_validation_data_segments'+'.csv')


    input_df = df
    # Tokenize the data using RoBERTa tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    class CustomDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            # Encode the text and ensure it has the same length by padding or truncating
            input_ids = tokenizer.encode(self.texts.iloc[idx], add_special_tokens=True, padding='max_length', max_length=256, truncation=True)
            return {"input_ids": torch.tensor(input_ids)}  # Convert to tensor

    # Create the evaluation dataset
    eval_dataset = CustomDataset(input_df['Segment_Text'])

    # DataLoaders
    eval_dataloader = DataLoader(eval_dataset, batch_size=8)

    # Load the pre-trained and fine-tuned model
    model_path = "./Roberta_Classification_Model_Fine_Tuned"  # Replace with your actual path
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Move the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Lists to store predictions and confidence scores
    predictions = []
    confidence_scores = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Extract input_ids from the batch
            input_ids = batch["input_ids"].to(device)

            # Get model outputs
            outputs = model(input_ids)
            logits = outputs.logits

            # Calculate softmax to get probabilities
            probabilities = torch.softmax(logits, dim=1)

            # Get predicted labels and confidence scores
            predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
            confidence = probabilities[:, 1].cpu().numpy()

            predictions.extend(predicted_labels)
            confidence_scores.extend(confidence)

    # Create a new dataframe with the original data and the model predictions
    df_evaluated = input_df.copy()
    df_evaluated['Predicted_Label'] = predictions
    df_evaluated['Confidence_Score'] = confidence_scores


    # csv_file_name = './evaluated_results__test_reports_from_complete_model.csv'
    # df_evaluated.to_csv(csv_file_name, index=False)
    evaluated_results_df = df_evaluated
    # evaluated_results_df =  pd.read_csv(csv_file_name)


    evaluated_results_df['Predicted_Label*Confidence_Score'] = evaluated_results_df['Predicted_Label'] * evaluated_results_df['Confidence_Score']
    # Initialize an empty DataFrame to store the final results
    final_summary_df = pd.DataFrame(columns=['Report_File_Path', 'generated_summary'])

    # Iterate over unique Report_File_Path values
    for report_file_path in evaluated_results_df['Report_File_Path'].unique():
        # Select rows for the current Report_File_Path
        subset_df = evaluated_results_df[evaluated_results_df['Report_File_Path'] == report_file_path].copy()

        # Calculate the rolling sum of the product over 4 consecutive rows
        subset_df['Rolling_Sum'] = subset_df['Predicted_Label*Confidence_Score'].rolling(window=4).sum()
        # Replace NaN values in the rolling sum with 0
        subset_df['Rolling_Sum'] = subset_df['Rolling_Sum'].fillna(0)
        # Find the starting index of the 4-row window with the maximum sum
        start_index = subset_df['Rolling_Sum'].idxmax()
        summary_file_path = subset_df['Summary_File_Path'].unique()[0]
        # Extract the top 4 consecutive rows
        top_4_rows = subset_df.loc[start_index:start_index + 3]
        list_segments = [i for i in range(start_index, start_index+4 )]
        # Concatenate the results to the final summary DataFrame
        generated_summary = ' '.join(top_4_rows['segments_for_summary'].tolist())
        generated_summary = '. '.join(generated_summary.split('. ')[1:-1])
        generated_summary += '.'
        final_summary_df = pd.concat([final_summary_df, pd.DataFrame({'Report_File_Path': [report_file_path], 'Summary_File_Path':[summary_file_path],'generated_summary': [generated_summary], 'list_segments': [list_segments]})])

    # Reset the index of the resulting DataFrame
    final_summary_df.reset_index(drop=True, inplace=True)

    #final_summary_df.to_csv('./Solution1_Validation_Results.csv')

    # Iterate over rows in the DataFrame
    for index, row in final_summary_df.iterrows():
        # Extract the generated summary and report file path
        generated_summary = row['generated_summary']
        report_file_path = row['Report_File_Path']
        summary_file_path = row['Summary_File_Path']
        # new filename as specified in ROUGE ReadMe
        file_name = summary_file_path
        # Write the generated summary to the text file with the new filename
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(generated_summary)

    print("summary files are created.")
    






