# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 10:30:08 2021


@author: akatz4
"""


"""

**Utility functions for embedding and clustering**

"""



import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import umap
#import umap.plot
import hdbscan




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import NMF, LatentDirichletAllocation, PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("sklearn modules imported")

#import torch


from sentence_transformers import SentenceTransformer
print("transformer module imported")

#import sys
#print(sys.path)




# At some point these warning messages started appearing:
    
    
# \Users\akatz4\AppData\Roaming\Python\Python37\site-packages\transformers\modeling_auto.py:837: FutureWarning: The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use `AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and `AutoModelForSeq2SeqLM` for encoder-decoder models.

# FutureWarning,
# \Users\akatz4\AppData\Roaming\Python\Python37\site-packages\transformers\tokenization_t5.py:184: UserWarning: This sequence already has </s>. In future versions this behavior may lead to duplicated eos tokens being added.
#   This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated eos tokens being added.







"""


huggingface transformer models

"""

# =============================================================================
# fine-grained sentiment analysis
# =============================================================================
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
classifier('This is a very good book.')

classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")


# =============================================================================
# emotion detection
# =============================================================================

from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")

model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")

"""
got this warning message:
The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. 
Please use `AutoModelForCausalLM` for causal language models, 
`AutoModelForMaskedLM` for masked language models 
and `AutoModelForSeq2SeqLM` for encoder-decoder models.

"""


def get_emotion(text):
  input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')

  output = model.generate(input_ids=input_ids,
               max_length=2)

  dec = [tokenizer.decode(ids) for ids in output]
  label = dec[0]
  return label


get_emotion("i feel as if i havent blogged in ages are at least truly blogged i am doing an update cute") # Output: 'joy'

get_emotion("i have a feeling i kinda lost my book") # Output: 'sadness'

get_emotion("this teacher was pretty bad") #Output: anger

get_emotion("this teacher was not very good") #Output: anger

get_emotion("nothing") #output: anger








# =============================================================================
# Original embedding
# =============================================================================

def embed_raw_text(raw_text, model_name, max_seq_length=128):
    """

    Parameters
    ----------
    raw_text : list
        List of the raw text where each element of the list is a string.
    model_name : str
        name of the model to use for embedding the text in higher dimensional space
        Can be one of ("bert_med", "bert_large", "roberta", "mpnet")
    max_seq_length : int
        max sequence length to pass to the transformer. Default value is 128 tokens.
    
    Returns
    -------
    embedding matrix of size {len(raw_text)} x {embedding-dim}

    """
    #(full list of models from https://github.com/UKPLab/sentence-transformers
    # medium model
    # medium model is 405 MB
    model_provided = False
    if model_name == 'mpnet':
        # embedder = SentenceTransformer('stsb-mpnet-base-v2')
        embedder = SentenceTransformer('paraphrase-mpnet-base-v2')
        print(f"Original model max_seq_length: {embedder.max_seq_length}.")
        embedder.max_seq_length=max_seq_length
        
        print(f"New model max_seq_length: {embedder.max_seq_length}.")
        model_provided = True
        
        
    elif model_name == 'all-mpnet':
        embedder = SentenceTransformer('all-mpnet-base-v2')
        print(f"Original model max_seq_length: {embedder.max_seq_length}.")
        embedder.max_seq_length=max_seq_length
        
        print(f"New model max_seq_length: {embedder.max_seq_length}.")
        model_provided = True
    
    
    elif model_name == 'bert_med':
        embedder = SentenceTransformer('bert-base-nli-mean-tokens')
        print(f"Original model max_seq_length: {embedder.max_seq_length}.")
        embedder.max_seq_length=max_seq_length
        
        print(f"New model max_seq_length: {embedder.max_seq_length}.")
        model_provided = True

    elif model_name == 'bert_large':
        # large model 
        # large model is 1.24 GB
        embedder = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
        print(f"Original model max_seq_length: {embedder.max_seq_length}.")
        embedder.max_seq_length=max_seq_length
        
        print(f"New model max_seq_length: {embedder.max_seq_length}.")
        model_provided = True

    elif model_name == 'roberta':        
        # large RoBERTa model
        # model size: 1.31 GB
        embedder = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
        print(f"Original model max_seq_length: {embedder.max_seq_length}.")
        embedder.max_seq_length=max_seq_length
        
        print(f"New model max_seq_length: {embedder.max_seq_length}.")
        model_provided = True

    elif model_name == 'all-miniLM':        
        # large RoBERTa model
        # model size: 1.31 GB
        embedder = SentenceTransformer('all-MiniLM-L12-v2')
        print(f"Original model max_seq_length: {embedder.max_seq_length}.")
        embedder.max_seq_length=max_seq_length
        
        print(f"New model max_seq_length: {embedder.max_seq_length}.")
        model_provided = True
        
    elif model_name == 't5':        
        # t5 (text-to-text transfer transformer model
        # model size: 1.31 GB
        embedder = SentenceTransformer('t5-large')
        print(f"Original model max_seq_length: {embedder.max_seq_length}.")
        embedder.max_seq_length=max_seq_length
        
        print(f"New model max_seq_length: {embedder.max_seq_length}.")
        model_provided = True

    else:
        print("Model name not recognized.")

    if model_provided:
        # embedding the titles
        corpus_embeddings = embedder.encode(raw_text)

        print("Raw text embedding completed.")
        print(f"Original raw text had dimension: {len(raw_text)}.")
        print(f"The dimension of the embedding is {corpus_embeddings.shape}.")

    return corpus_embeddings        


#test_embed = embed_raw_text(titles_list, 'bert_med')    
        
#del test_embed    

# =============================================================================
# # dimension reduction utility functions
# =============================================================================

def project_original_embedding(original_embedding, embed_param_dict, 
                               high_to_mid_method='pca', mid_to_low_method='umap',
                               to_low=True, plot_pca=True, title=''):
    if high_to_mid_method == 'pca':
        pca_dim = embed_param_dict['pca_dim']
        pca = PCA(n_components=pca_dim, random_state=42)
        projected_med = pca.fit_transform(original_embedding)
        
        if plot_pca:
            plt.plot(np.cumsum(pca.fit(original_embedding).explained_variance_ratio_))
            plt.xlabel('Number of components')
            plt.ylabel('Cumulative explained variance');
        
    if mid_to_low_method == 'umap':
        projected_low = project_draw_umap(projected_med, 
                                          n_neighbors=embed_param_dict['n_neighbors'],
                                          min_dist=embed_param_dict['min_dist'],
                                          n_components=embed_param_dict['n_components'],
                                          metric=embed_param_dict['metric'],
                                          random_state=embed_param_dict['random_state'],
                                          title=title)
    
    if mid_to_low_method == 'tsne':
        projected_low = project_draw_tsne(projected_med)
    
   
    if to_low:
        projected_final = projected_low
    else:
        projected_final = projected_med

    return projected_final



def project_umap(data, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=123):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state
    )
    u = fit.fit_transform(data);
    print(f"UMAP projection completed. Started with {data.shape}. Returning {u.shape}.")
    return u



def draw_umap(umap_projection, title=''):
    print(f"Dimension of umap projection: {umap_projection.shape}.")
    fig = plt.figure()
    n_components = umap_projection.shape[1]
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(umap_projection[:,0], range(len(umap_projection)))
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(umap_projection[:,0], umap_projection[:,1])
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(umap_projection[:,0], umap_projection[:,1], umap_projection[:,2], s=100)
    plt.title(title, fontsize=12)
    

    
def project_draw_umap(input_data, n_neighbors=15, min_dist=0.1, 
                      n_components=2, metric='euclidean', random_state=123,
                      title=''):
    umap_embedding = project_umap(input_data, n_neighbors, min_dist, n_components, metric, random_state)
    
    draw_umap(umap_embedding, title=title)
    return umap_embedding




def project_tsne(input_data, n_components=2, perplexity = 30.0, n_iter=1000):
    fit = TSNE(n_components=n_components, 
               perplexity=perplexity, 
               n_iter=n_iter, 
               verbose=1)
    t = fit.fit_transform(input_data)
    print(f"t-sne projection completed. Started with {input_data.shape}. Returning {t.shape}.")
    return t
    
    
    
def draw_tsne(tsne_projection, title=''):
    print(f"Dimension of t-sne projection: {tsne_projection.shape}.")
    fig = plt.figure()
    n_components = tsne_projection.shape[1]
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(tsne_projection[:,0], range(len(tsne_projection)))
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(tsne_projection[:,0], tsne_projection[:,1])
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tsne_projection[:,0], tsne_projection[:,1], tsne_projection[:,2], s=100)
    plt.title(title, fontsize=12)
    
    
def project_draw_tsne(input_data):
    tsne_embedding = project_tsne(input_data)
    
    draw_tsne(tsne_embedding)
    return tsne_embedding


# =============================================================================
# Clustering utility functions
# =============================================================================

def cluster_printing(original_corpus_list, clustering_model):
    ### Universal method for storing cluster assignments and sentences in clusters
    cluster_assignment = clustering_model.labels_
    print(cluster_assignment)
    
    clustered_sentences = [[] for i in range(len(np.unique(clustering_model.labels_)))]
    clustered_sentences
    
    print(f"The minimum cluster number is: {min(clustering_model.labels_)}.")
    print(f"The length of cluster_model.labels_ is {len(clustering_model.labels_)}.")
    print(f"The length of the original corpus list is {len(original_corpus_list)}.")
    if min(clustering_model.labels_) < 0:        
        ### Method for handling -1 cluster assignment from HDBSCAN
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            #print(f"cluster id is: {cluster_id}")
            #print(f"sentence id is: {sentence_id}")
            clustered_sentences[cluster_id+1].append(original_corpus_list[sentence_id])
                
        for i, cluster in enumerate(clustered_sentences):
            print("Cluster ", i-1)
            print("Cluster size:", len(cluster))
            print(cluster)
            print("")
            
    else:    
        ### Method for regular clustering assignments that start at 0
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[cluster_id].append(original_corpus_list[sentence_id])
            
        
        for i, cluster in enumerate(clustered_sentences):
            print("Cluster ", i)
            print("Cluster size:", len(cluster))
            print(cluster)
            print("")
        



def cluster_embedding(data, param_dict, original_corpus_list=None, model='hdbscan', plot_option=False, plot_dist=True, print_clust=True):
    print(f"You have selected {model} for clustering.")
    if model == 'hdbscan':        
        ### use HDBSCAN for clustering instead of K-means
        clustering_model = hdbscan.HDBSCAN(min_cluster_size=param_dict['min_cluster_size'],
                                           min_samples=param_dict['min_samples'],
                                           cluster_selection_epsilon=param_dict['cluster_selection_epsilon'],
                                           metric=param_dict['metric'])

    if model == 'kmeans':
        ### Use Kmean clustering on embeddings
        num_clusters = param_dict['num_clusters']
        clustering_model = KMeans(n_clusters=num_clusters)

    if model == 'agglomerative':
        ### Use Agglomerative clustering on embeddings
        if param_dict['agg_type'] == "threshold":
            clustering_model = AgglomerativeClustering(n_clusters=None, 
                                                       distance_threshold=param_dict['threshold_val']) #, affinity='cosine', linkage='average', distance_threshold=0.4)
            # clustering_model.fit(corpus_embeddings)
            # cluster_assignment = clustering_model.labels_

        
        if param_dict['agg_type'] == "n_cluster":
            clustering_model = AgglomerativeClustering(linkage=param_dict['linkage'],
                                                       n_clusters=param_dict['n_clusters'],
                                                       affinity=param_dict['affinity'])
        
    clustering_model.fit(data)
    print("Clustering completed.")
    
    
    if plot_dist == True:
        ### histogram of cluster assignments
        plt.figure(1)
        sns.displot(clustering_model.labels_, kde=False, rug=True, bins = len(np.unique(clustering_model.labels_)))
        

    if plot_option == True and data.shape[1] == 2:
        ### plotting the results of dim reduction and clustering together
        plt.figure(2)
        plt.scatter(data[:, 0], data[:, 1],
                    c=clustering_model.labels_, edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('Spectral', len(np.unique(clustering_model.labels_))))
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.colorbar()
    
    plt.show()

    if print_clust == True:
        cluster_printing(original_corpus_list=original_corpus_list, clustering_model=clustering_model)
    
    cluster_result = clustering_model        
    return cluster_result






"""

from top2vec article: https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6

"""

"""
docs_df = pd.DataFrame(data, columns=["Doc"])
docs_df['Topic'] = cluster.labels_
docs_df['Doc_ID'] = range(len(docs_df))
docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count
  
tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(data))



def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes

top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(10)

"""