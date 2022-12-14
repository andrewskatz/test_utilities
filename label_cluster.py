# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 08:43:50 2022

@author: akatz4
"""



"""

Utility functions for labeling responses using an example bank

"""



import os
import sys
import pandas as pd


from personal_utilities import embed_cluster as ec

import pickle
import numpy as np
import re

#import umap.plot

from sklearn.manifold import MDS, TSNE

from textblob import TextBlob
import nltk
nltk.download('punkt')


import pickle

from scipy.spatial.distance import cdist


import nltk
from sentence_transformers import SentenceTransformer, util
import numpy as np
#from lexrank import LexRank
#from lexrank.lexrank import degree_centrality_scores
import math




def create_lexrank_summary_of_clusters(clustered_df,
                                       id_col, 
                                       cluster_label_col='cluster_label', 
                                       text_col = 'split_sent',
                                       lexrank_sum_size=0.2,
                                       transformer_model='all-MiniLM-L12-v2'):
  
  """
  Function to create lexrank summaries for each cluster returned from cluster labeling
  Parameters
  --------
    clustered_df: dataframe
    id_col: str
    cluster_label_col: str
    text_col: str
    lexrank_sum_size: double (0,1)
    transformer_model: str



  Returns
  --------
    lexrank_summary_df: dataframe with extractive summary sentences 
    bad_clusters: list of clusters that through an exception for some reason
  
  """

  lexrank_summary_dict = {
      'sent_id': [],
      'cluster_label':[],
      'lex_sum_sent':[]
  }

  bad_clusters=[]

  # create the sentence transformer model object
  model = SentenceTransformer(transformer_model)



  for i in np.sort(clustered_df[cluster_label_col].unique()):
      try:
        print(f"\nWorking on cluster {i}")
        temp_cl_df = clustered_df[clustered_df[cluster_label_col] == i]
        
        # decided not to use previously calculated embeddings because they would need to be converted to tensor
        # test_embed = test_df.loc[:,'0':'74']
        
        
        sentences = temp_cl_df[text_col].to_list()
        
        print(f"There are {len(sentences)} sentences in cluster {i}.")

        embeddings = model.encode(sentences, convert_to_tensor=True)
        # embeddings = test_embed
        
        print(f"Finished embedding for cluster {i}.")
        
        
        #Compute the pair-wise cosine similarities
        cos_scores = util.pytorch_cos_sim(embeddings, embeddings).numpy()
        
        print(f"Calculated cosine similarities for cluster {i}.")
        #Compute the centrality for each sentence
        centrality_scores = degree_centrality_scores(cos_scores, threshold=None)
        
        print(f"Completed centrality scores for cluster {i}.")
        #We argsort so that the first element is the sentence with the highest score
        most_central_sentence_indices = np.argsort(-centrality_scores)
        cl_sent_num = len(sentences)
        sum_sent_cap = int(math.ceil(cl_sent_num * lexrank_sum_size))
        
        #Print the top 20% of sentences with the highest scores
        print(f"Summary for cluster {i}: ")
        for idx in most_central_sentence_indices[0:sum_sent_cap]:
            print(sentences[idx].strip())
            
            lexrank_summary_dict['sent_id'].append(temp_cl_df[id_col].iloc[idx])
            lexrank_summary_dict['cluster_label'].append(i)
            lexrank_summary_dict['sum_sent'].append(sentences[idx].strip())
      
      except:
        bad_clusters.append(i)

  lexrank_summary_df = pd.DataFrame(lexrank_summary_dict)

  return lexrank_summary_df, bad_clusters






def build_example_bank(lexrank_summary_df, codebook_df, text_col='text_col', keep_n=2):
  """
  Function to create the example bank by taking lexrank summary and combining with the consolidated codebook to add example_label column

  Parameters
  --------
    lexrank_summary_df: dataframe
    codebook_df: dataframe
      should have column for 

  Returns
  --------
    example_df: dataframe
  """
  # example_df should have a column called manual_text_label_v1 from the initial labeling of lexrank_summary_df
  # codebook_df should have a column for manual_text_label_v1 and then example_label from consolidating the manual text labels
  lexrank_summary_df = lexrank_summary_df.dropna()
  lexrank_summary_df = lexrank_summary_df.groupby(text_col).head(keep_n) # only keep the first n instances of a comment
  example_df = pd.merge(lexrank_summary_df, codebook_df, on='manual_text_label_v1', how = 'left')
  example_df = example_df.drop(columns=['manual_text_label_v1'])
  

  return example_df










def label_the_unlabeled(example_df,
                        unlabeled_df,
                        full_embeddings=None,
                        need_to_embed=False,
                        unlabeled_id_col = 'sent_id',
                        unlabeled_text_col='text_col',
                        unlabeled_cluster_label_col='cluster_label',
                        example_id_col='sent_id',
                        example_text_col='text_col',
                        example_label_col='example_label',
                        embedding_model = 'all-miniLM',
                        similarity_type='count',
                        sim_n=5):
  """
  Function to label the unlabeled sentences in a corpus. Requires a codebook of labeled possibilities

  Parameters
  --------
    example_df: dataframe [n x 3]
      dataframe with n labeled examples, cols are example_label, example_text, and example_id
    full_embeddings: 2D numpy array [(n+m) x d]
      d-dimensional embeddings of the full dataset
    unlabeled_df: dataframe [m x p]
      m x p dimensional array with m unlabeled observations
    unlabeled_id_col: str
      column with sentence ids
    unlabeled_text_col: str
      column with the text to be labeled
    unlabeled_cluster_label_col: str
      column with the cluster (number) labels
    example_id_col: str
      column with sentence ids in the example_df
    example_text_col: str
      column with the labeled example text
    example_label_col: str
      column with the example labels
    embedding_model: str
      type of sentence transformer model to use for the embeddings
    similarity_type: str
      option to use a cosine similarity threshold or count as the way to identify similar labels
    sim_n: int
      number of similar instances to find

  Returns
  --------
    labeled_df: dataframe

    bad_clusters: list
      clusters that threw an error during labeling

  """
  

  sim_ex_dict = {'sent_id':[],
                 'original_sent_text':[],
                 'original_cluster_label':[],
                 'similarity_rank':[],
                 'similarity_score':[],
                 'similar_ex_id':[],
                 'similar_ex_text':[],
                 'similar_ex_label':[]}


  bad_clusters=[]
  
  example_id_list = example_df[example_id_col].to_list()
  
  if need_to_embed:
      example_embeddings = ec.embed_raw_text(example_df[example_text_col].to_list(), embedding_model, max_seq_length=200)
  else:
      example_embeddings = full_embeddings[example_id_list]

  for i in np.sort(unlabeled_df[unlabeled_cluster_label_col].unique()):
  #for i in range(10):
    try:
      print(f"\nWorking on cluster {i}")
      temp_cl_df = unlabeled_df[unlabeled_df[unlabeled_cluster_label_col] == i]

      unlabeled_text_list = temp_cl_df[unlabeled_text_col].to_list()
      unlabeled_id_list = temp_cl_df[unlabeled_id_col].to_list()
      
      if need_to_embed:
          unlabeled_embeddings = ec.embed_raw_text(unlabeled_text_list, embedding_model, max_seq_length=200)
      else:
          unlabeled_embeddings = full_embeddings[unlabeled_id_list]

      print(unlabeled_embeddings.shape)

      print("Calculating similarity scores")
      cosine_df = 1 - cdist(unlabeled_embeddings, example_embeddings, metric='cosine')

      scores_df = pd.DataFrame(cosine_df)

        # differences between similarity_type ('count' vs 'top_score')
      if similarity_type == "count":
          
          for j in range(scores_df.shape[0]):
            top_n_scores = scores_df.iloc[j,:].nlargest(sim_n)
            #print(top_n_scores)
    
    
            indices = top_n_scores.index
            #print(indices)
    
    
            for num, k in enumerate(indices):
              # print(num)
              # print(i)
              # print(f"Item {j} had score {top_n_scores.iloc[num]}. The task was: {tasks_text[j]}")
              # print(f"The original classification of this task was {tasks_df['label'].iloc[j]}")
    
              #print(f"Cluster label: {temp_cl_df['cluster_label'].iloc[j]}")
              #print(labeled_sent_df['sum_sent'].iloc[k])
              #print(temp_cl_df['split_sent'].iloc[j])
    
              sim_ex_dict['sent_id'].append(temp_cl_df[unlabeled_id_col].iloc[j])
              sim_ex_dict['original_sent_text'].append(temp_cl_df[unlabeled_text_col].iloc[j])
              sim_ex_dict['original_cluster_label'].append(temp_cl_df[unlabeled_cluster_label_col].iloc[j])
              sim_ex_dict['similarity_rank'].append(num)
              sim_ex_dict['similarity_score'].append(top_n_scores.iloc[num])
              sim_ex_dict['similar_ex_id'].append(example_df[example_id_col].iloc[k])
              sim_ex_dict['similar_ex_text'].append(example_df[example_text_col].iloc[k])
              sim_ex_dict['similar_ex_label'].append(example_df[example_label_col].iloc[k])
    
      elif similarity_type == "top_score":

          

          sim_n = 1 # just pick the top score



          
          for j in range(scores_df.shape[0]):
            top_n_scores = scores_df.iloc[j,:].nlargest(sim_n)
            # print(f'the top n scores are {top_n_scores}')
    
    
            indices = top_n_scores.index
            # print(f'the indices are {indices}')
    
    
            for num, k in enumerate(indices):
              # print(num)
              # print(k)
              # print(f"Item {j} had score {top_n_scores.iloc[num]}. ")
              # print(f"The original classification of this task was {tasks_df['label'].iloc[j]}")
    
              # print(f"Unlabeled text: {temp_cl_df['split_sent'].iloc[j]}")
              # print(f'labeled example text: {example_df["split_sent"].iloc[k]}')
              # print(temp_cl_df['split_sent'].iloc[j])
    
              sim_ex_dict['sent_id'].append(temp_cl_df[unlabeled_id_col].iloc[j])
              sim_ex_dict['original_sent_text'].append(temp_cl_df[unlabeled_text_col].iloc[j])
              sim_ex_dict['original_cluster_label'].append(temp_cl_df[unlabeled_cluster_label_col].iloc[j])
              sim_ex_dict['similarity_rank'].append(num)
              sim_ex_dict['similarity_score'].append(top_n_scores.iloc[num])
              sim_ex_dict['similar_ex_id'].append(example_df[example_id_col].iloc[k])
              sim_ex_dict['similar_ex_text'].append(example_df[example_text_col].iloc[k])
              sim_ex_dict['similar_ex_label'].append(example_df[example_label_col].iloc[k])
          
            
            


    except:
      print(f"Error encountered with cluster {i}")
      bad_clusters.append(i)


  # print(f' the length of the dict list is: {len(sim_ex_dict["sent_id"])}')  
  
  sim_ex_df = pd.DataFrame(sim_ex_dict)

  return sim_ex_df, bad_clusters












def get_sim_counts(sim_ex_df, id_col='sent_id', similar_ex_label_col='similar_ex_label', print_headers=False):

  """
  Function to calculate the votes for each label for each observation

  Parameters
  --------
    sim_ex_df: dataframe
    id_col: str
    similar_ex_label: str
    print_header: bool  

  Returns
  --------
    counts_df: dataframe
      dataframe with the counts for each label for each observation
  """
  
  print(f"Started with {sim_ex_df[id_col].nunique()} observations.")

  counts_df = sim_ex_df.groupby([id_col, similar_ex_label_col])[similar_ex_label_col].agg('count').reset_index(name='label_counts')
  print(f"Counts dataframe has shape {counts_df.shape}.")

  if print_headers:
    print(f"The column headers are: {counts_df.columns}")



  return counts_df









def get_majority(counts_df, id_col='sent_id', counts_col='label_counts', n=5):

  """
  Function to find which observations have a majority vote for one label

  Parameters
  --------
    counts_df: dataframe
    id_col: str
    counts_col: strr
    n: int

  Returns
  --------
    has_majority_df: dataframe
    has_no_majority_df: dataframe
  
  """
  n = int(n)
  unique_n = counts_df[id_col].nunique()
  print(f"Started with {unique_n} observations.")

  has_majority_df = counts_df[counts_df[counts_col] > n/2]
  has_majority_ids = has_majority_df[id_col].to_list()
  print(f"There are {len(has_majority_ids)} ids in the majority_df")

  has_no_majority_df = counts_df[~counts_df[id_col].isin(has_majority_ids)] 
  # need to drop duplicates from has_no_majority_df
  has_no_majority_df = has_no_majority_df.drop_duplicates(subset=[id_col])

  majority_n = has_majority_df[id_col].nunique()
  no_majority_n = has_no_majority_df[id_col].nunique()

  print(f"Using {n/2} as the threshold, there are {majority_n} with a majority vote.")
  print(f"Sanity check: there are {no_majority_n} without a majority vote.")

  return has_majority_df, has_no_majority_df









def label_and_filter(example_df, 
                     unlabeled_df,
                     full_embeddings='',
                     need_to_embed=False,
                     unlabeled_id_col = 'sent_id',
                     unlabeled_text_col='split_sent',
                     unlabeled_cluster_label_col='cluster_label',
                     example_id_col='sent_id',
                     example_text_col='sum_sent',
                     example_label_col='example_label',
                     embedding_model = 'all-miniLM',
                     similarity_type='count',
                     sim_n=5,
                     labeling_param_dict={},
                     counts_param_dict={},
                     majority_param_dict={}):
  
  """
  Function to label, get counts, and get majority

  Parameters
  --------
    example_df: dataframe [n x 3]
      dataframe with n labeled examples, cols are example_label, example_text, and example_id
    example_embeddings: 2D numpy array [n x d]
      d-dimensional embeddings of the codebook_df labeled examples
    unlabeled_df: dataframe [m x p]
      m x p dimensional array with m unlabeled observations
    param_dict:
      unlabeled_id_col: str
        column with sentence ids
      unlabeled_text_col: str
        column with the text to be labeled
      unlabeled_cluster_label_col: str
        column with the cluster (number) labels
      example_id_col: str
        column with sentence ids in the example_df
      example_text_col: str
        column with the labeled example text
      example_label_col: str
        column with the example labels
      embedding_model: str
        type of sentence transformer model to use for the embeddings
      similarity_type: str
        option to use a cosine similarity threshold or count as the way to identify similar labels
      sim_n: int
        number of similar instances to find
      labeling_param_dict: dictionary
        to override the labeling parameters in label_the_unlabeled()
      counts_param_dict: dictionary
        to override default parameters in get_counts()
      majority_param_dict: dictionary
        to override default parameters in get_majority()

  Returns
  --------
    has_majority_df: dataframe
    has_no_majority_df: dataframe
  
  
  """
  # check to see if param dictionary provided to override defaults for label_the_unlabeled()
  if labeling_param_dict:
    sim_ex_df, bad_clusters = label_the_unlabeled(example_df, 
                                    unlabeled_df,
                                    full_embeddings='',
                                    need_to_embed=need_to_embed,
                                    unlabeled_id_col = labeling_param_dict['unlabeled_id_col'],
                                    unlabeled_text_col= labeling_param_dict['unlabeled_text_col'],
                                    unlabeled_cluster_label_col = labeling_param_dict['unlabeled_cluster_label_col'],
                                    example_id_col = labeling_param_dict['example_id_col'],
                                    example_text_col= labeling_param_dict['example_text_col'],
                                    example_label_col = labeling_param_dict['example_label_col'],
                                    embedding_model = labeling_param_dict['embedding_model'],
                                    similarity_type = labeling_param_dict['similarity_type'],
                                    sim_n = labeling_param_dict['sim_n'])
  else:
    sim_ex_df, bad_clusters = label_the_unlabeled(example_df,  
                                    unlabeled_df,
                                    full_embeddings='',
                                    need_to_embed=need_to_embed)


  print("Finished labeling the unlabeled sentences.")
  print(f"sim_ex_df has shape {sim_ex_df.shape}.")
  print(f"sim_ex_df has columns {sim_ex_df.columns}")

  # check to see if param dictionary provided to override defaults for get_counts()
  if counts_param_dict:
    print_header = counts_param_dict['print_header']
    counts_df = get_sim_counts(sim_ex_df, print_header = print_header)
  else:
    counts_df = get_sim_counts(sim_ex_df)




  print("Finished getting counts.")

  # check to see if param dictionary to override the defaults for get_majority()
  if majority_param_dict:
    counts_id_col = counts_param_dict['id_col']
    counts_n = counts_param_dict['n']
    has_majority_df, has_no_majority_df = get_majority(counts_df,
                                                       id_col=counts_id_col,
                                                       n = counts_n)    
  else:
    has_majority_df, has_no_majority_df = get_majority(counts_df)



  print("Finished finding observations with majority vote.")

  return has_majority_df, has_no_majority_df







def grow_example_bank(example_df, unlabeled_df, full_embeddings = '', need_to_embed=False, param_dict={}):

  """
  Function to grow the example bank by adding example_df to has_majority_df (created inside the function using label_and_filter())

  Parameters
  --------
    example_df: dataframe
      dataframe to use for comparison
    example_embeddings: numpy array
    unlabeled_df: dataframe

  Returns
  --------
    bigger_ex_bank_df: dataframe
    has_no_majority_df: dataframe
    found_more_majority_flag: bool

  """
  has_majority_df, has_no_majority_df = label_and_filter(example_df, 
                                                         unlabeled_df,
                                                         full_embeddings,
                                                         need_to_embed=need_to_embed)
  
  # add the has_majority_df observations to the example_df
  has_majority_n = has_majority_df.shape[0]
  if has_majority_n > 0:
    found_more_majority_flag = True
  else:
    found_more_majority_flag = False

  has_no_majority_n = has_no_majority_df.shape[0]
  if has_no_majority_n == 0:
    no_more_to_label_flag = True
  else:
    no_more_to_label_flag = False

  has_majority_df = pd.merge(has_majority_df, unlabeled_df, on = 'sent_id', how='left')

  has_no_majority_df = pd.merge(has_no_majority_df, unlabeled_df, on = 'sent_id', how='left')


  has_majority_df.rename(columns={'similar_ex_label':'example_label'})

  bigger_ex_bank_df = pd.concat([example_df, has_majority_df])

  print(f"Started with {example_df.shape[0]} observations in starting example bank and ended with {bigger_ex_bank_df.shape[0]} observations in new example bank")

  return bigger_ex_bank_df, has_no_majority_df, found_more_majority_flag, no_more_to_label_flag








def check_condition(found_more_majority_flag=False, no_more_to_label_flag=True):
  """
  Function to check if there are more items to label or whether the labeling has reached a stalemate

  Parameters
  --------
    found_more_majority_flag: bool
      True if more have been labeled, otherwise false
    no_more_to_label_flag: bool
      True if there are no more to label, otherwise false

  Returns
  --------
    keep_looping: bool

  """

  keep_looping = True
  if no_more_to_label_flag or not found_more_majority_flag:
    keep_looping = False

  return keep_looping
  














"""
LexRank implementation
Source: https://github.com/crabcamp/lexrank/tree/dev
"""

import numpy as np
from scipy.sparse.csgraph import connected_components

def degree_centrality_scores(
    similarity_matrix,
    threshold=None,
    increase_power=True,
):
    if not (
        threshold is None
        or isinstance(threshold, float)
        and 0 <= threshold < 1
    ):
        raise ValueError(
            '\'threshold\' should be a floating-point number '
            'from the interval [0, 1) or None',
        )

    if threshold is None:
        markov_matrix = create_markov_matrix(similarity_matrix)

    else:
        markov_matrix = create_markov_matrix_discrete(
            similarity_matrix,
            threshold,
        )

    scores = stationary_distribution(
        markov_matrix,
        increase_power=increase_power,
        normalized=False,
    )

    return scores


def _power_method(transition_matrix, increase_power=True):
    eigenvector = np.ones(len(transition_matrix))

    if len(eigenvector) == 1:
        return eigenvector

    transition = transition_matrix.transpose()

    while True:
        eigenvector_next = np.dot(transition, eigenvector)

        if np.allclose(eigenvector_next, eigenvector):
            return eigenvector_next

        eigenvector = eigenvector_next

        if increase_power:
            transition = np.dot(transition, transition)


def connected_nodes(matrix):
    _, labels = connected_components(matrix)

    groups = []

    for tag in np.unique(labels):
        group = np.where(labels == tag)[0]
        groups.append(group)

    return groups


def create_markov_matrix(weights_matrix):
    n_1, n_2 = weights_matrix.shape
    if n_1 != n_2:
        raise ValueError('\'weights_matrix\' should be square')

    row_sum = weights_matrix.sum(axis=1, keepdims=True)

    return weights_matrix / row_sum


def create_markov_matrix_discrete(weights_matrix, threshold):
    discrete_weights_matrix = np.zeros(weights_matrix.shape)
    ixs = np.where(weights_matrix >= threshold)
    discrete_weights_matrix[ixs] = 1

    return create_markov_matrix(discrete_weights_matrix)


def graph_nodes_clusters(transition_matrix, increase_power=True):
    clusters = connected_nodes(transition_matrix)
    clusters.sort(key=len, reverse=True)

    centroid_scores = []

    for group in clusters:
        t_matrix = transition_matrix[np.ix_(group, group)]
        eigenvector = _power_method(t_matrix, increase_power=increase_power)
        centroid_scores.append(eigenvector / len(group))

    return clusters, centroid_scores


def stationary_distribution(
    transition_matrix,
    increase_power=True,
    normalized=True,
):
    n_1, n_2 = transition_matrix.shape
    if n_1 != n_2:
        raise ValueError('\'transition_matrix\' should be square')

    distribution = np.zeros(n_1)

    grouped_indices = connected_nodes(transition_matrix)

    for group in grouped_indices:
        t_matrix = transition_matrix[np.ix_(group, group)]
        eigenvector = _power_method(t_matrix, increase_power=increase_power)
        distribution[group] = eigenvector

    if normalized:
        distribution /= n_1

    return distribution





