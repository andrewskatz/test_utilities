# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 08:17:16 2022

@author: akatz4
"""

from transformers import pipeline
import pandas as pd

classifier = pipeline(task = 'zero-shot-classification', model = 'facebook/bart-large-mnli')


def label_df_with_zs(unlabeled_df, 
                     text_col, 
                     id_col, 
                     class_labels, 
                     zs_threshold, 
                     multi_label=False,
                     keep_top_n=False,
                     top_n=5):
    """Function to label item items in a dataframe in the text_col with a zero-shot classifier
    Args:
        unlabeled_df: dataframe
            dataframe to label
        text_col: string 
            name of text column
        id_col: str
            name of id column in unlabeled df
        class_labels: list
            list of labels to consider
        zs_threhsold: float
            number between 0 and 1
        multi_label: bool
            select whether to treat as single classification of multiclassification task
        keep_top_n: bool
            only applicable when multi_label= True; have option of keeping only the labels with highest probability
        top_n: int
            number of top matches to keep
    
    Returns:
        labeled dataframe with sequence, labels, scores, and new_sent_id
    
    
    """
    if multi_label == True:
        if keep_top_n == False:
            test_results_df = pd.DataFrame(columns=['sequence', 'labels', 'scores', 'original_id'])
            
            for index, row in unlabeled_df.iterrows():
              row_text = row[text_col]
              original_sent_id = row[id_col]
              print(f"working on item {index} with id {original_sent_id}: {row_text}")
              
              classifier_results = classifier(row_text, class_labels, multi_label=True)
              all_results_df = pd.DataFrame(classifier_results)
              all_results_df['original_id'] = original_sent_id
            
            
              final_results_df = all_results_df.head(1) # start by picking the top match to make sure at least one match returned
              cutoff_results_df = all_results_df[all_results_df['scores'] > zs_threshold] # now select all results above a threshold cutoff
            
              final_results_df = pd.concat([final_results_df, cutoff_results_df]) # combine the two together
              final_results_df = final_results_df.drop_duplicates() # drop the duplicated top entry
            
              test_results_df = pd.concat([test_results_df, final_results_df])
          

        if keep_top_n == True:
            test_results_df = pd.DataFrame(columns=['sequence', 'labels', 'scores', 'original_id'])
            
            for index, row in unlabeled_df.iterrows():
              row_text = row[text_col]
              original_sent_id = row[id_col]
              print(f"working on item {index} with id {original_sent_id}: {row_text}")
              
              classifier_results = classifier(row_text, class_labels, multi_label=True)
              all_results_df = pd.DataFrame(classifier_results)
              all_results_df['original_id'] = original_sent_id
            
            
              final_results_df = all_results_df.head(top_n) # start by picking the top match to make sure at least one match returned
              
            
              test_results_df = pd.concat([test_results_df, final_results_df])



          
          
          
    if multi_label == False:
        
        test_results_df = pd.DataFrame(columns=['sequence', 'labels', 'scores', 'original_id'])
        
        for index, row in unlabeled_df.iterrows():
          row_text = row[text_col]
          original_sent_id = row[id_col]
          print(f"working on item {index} with id {original_sent_id}: {row_text}")
          
          classifier_results = classifier(row_text, class_labels)
          all_results_df = pd.DataFrame(classifier_results)
          all_results_df['original_id'] = original_sent_id
        
        
          final_results_df = all_results_df.head(1) # start by picking the top match to make sure at least one match returned
          cutoff_results_df = all_results_df[all_results_df['scores'] > zs_threshold] # now select all results above a threshold cutoff
        
          final_results_df = pd.concat([final_results_df, cutoff_results_df]) # combine the two together
          final_results_df = final_results_df.drop_duplicates() # drop the duplicated top entry
        
          test_results_df = pd.concat([test_results_df, final_results_df])
      
    return test_results_df
