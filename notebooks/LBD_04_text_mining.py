#!/usr/bin/env python
# coding: utf-8

# # 4. LBD Text Mining Module
# 
# This module implements various text mining algorithms like topic modeling, sentiment analysis, and clustering.

# In[2]:


import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from scipy.sparse import csr_matrix
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import List, Tuple, Any


# This function perform_topic_modeling takes a document-term matrix in the Compressed Sparse Row (CSR) format and an integer 
# specifying the number of topics to extract as input, and uses the scikit-learn library to perform topic modeling using 
# Latent Dirichlet Allocation (LDA). 
# The function first creates an instance of the sklearn.decomposition.LatentDirichletAllocation class 
# with the specified number of components and a fixed random state of 42 for reproducibility.
# 
# The function then fits the LDA model to the input document-term matrix and transforms it into a document-topic matrix 
# using the lda.fit_transform() method.
# 
# Finally, the function returns a tuple containing the LDA model and the document-topic matrix as a NumPy array.
# 
# 

# In[ ]:


def perform_topic_modeling(matrix: csr_matrix, n_topics: int) -> Tuple[BaseEstimator, np.ndarray]:
    """
    Perform topic modeling on a document-term matrix using Latent Dirichlet Allocation (LDA).
    :param matrix: csr_matrix, the document-term matrix to perform topic modeling on
    :param n_topics: int, the number of topics to extract
    :return: Tuple[BaseEstimator, np.ndarray], the fitted topic modeling LDA model and the document-topic matrix
    """
    # Create an instance of the LatentDirichletAllocation class from scikit-learn
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)

    # Fit the LDA model to the document-term matrix and transform it into a document-topic matrix
    document_topic_matrix = lda.fit_transform(matrix)

    # Return the LDA model and the document-topic matrix as a tuple
    return lda, document_topic_matrix



# This function perform_sentiment_analysis takes a list of tokenized text documents as input and uses 
# the Natural Language Toolkit (NLTK) library to perform sentiment analysis using 
# the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool. 
# The function first ensures that the required NLTK data for the VADER sentiment analysis tool 
# is downloaded using the nltk.download() method with the quiet=True argument to suppress download messages.
# 
# The function then creates an instance of the nltk.sentiment.SentimentIntensityAnalyzer, 
# which is an implementation of the VADER sentiment analysis algorithm. 
# The function initializes an empty list sentiment_scores to store the sentiment scores for each document.
# 
# For each document in the input tokens_list, the function calculates the sentiment score by first joining 
# the tokens into a single string using the join() method. The function then calculates the sentiment score for the document 
# using the SentimentIntensityAnalyzer's polarity_scores() method, which returns a dictionary containing 
# the sentiment scores for the positive, negative, neutral, and compound sentiment components. 
# The function appends the compound sentiment score to the sentiment_scores list.
# 
# Finally, the function returns the list of sentiment scores.
# 
# 

# In[ ]:


def perform_sentiment_analysis(tokens_list: List[List[str]]) -> List[float]:
    """
    Perform sentiment analysis on a list of tokenized text documents using the VADER sentiment analysis tool from NLTK.
    :param tokens_list: List[List[str]], a list of tokenized text documents
    :return: List[float], a list of sentiment scores for each document
    """
    # Ensure the NLTK data is downloaded
    nltk.download('vader_lexicon', quiet=True)
    
    # Create an instance of the SentimentIntensityAnalyzer from NLTK
    sia = SentimentIntensityAnalyzer()

    # Initialize an empty list to store the sentiment scores
    sentiment_scores = []

    # Calculate the sentiment scores for each document in the tokens_list
    for tokens in tokens_list:
        # Join the tokens into a single string
        document = ' '.join(tokens)
        
        # Calculate the sentiment score using the SentimentIntensityAnalyzer
        sentiment = sia.polarity_scores(document)
        
        # Append the compound sentiment score to the sentiment_scores list
        sentiment_scores.append(sentiment['compound'])

    return sentiment_scores



# This function perform_clustering takes a document-term matrix in the Compressed Sparse Row (CSR) format and 
# an integer specifying the number of clusters to create as input, and uses the scikit-learn library 
# to perform clustering using the K-Means algorithm. The function first creates an instance 
# of the sklearn.cluster.KMeans class with the specified number of clusters and a fixed random state of 42 for reproducibility.
# 
# The function then fits the K-Means model to the input document-term matrix using the kmeans.fit() method. 
# After fitting the model, the function obtains the cluster labels for each document using the kmeans.labels_ attribute.
# 
# Finally, the function returns a tuple containing the K-Means model and the cluster labels as a NumPy array.
# 
# 

# In[ ]:


def perform_clustering(matrix: csr_matrix, n_clusters: int) -> Tuple[BaseEstimator, np.ndarray]:
    """
    Perform clustering on a document-term matrix using K-Means.
    :param matrix: csr_matrix, the document-term matrix to perform clustering on
    :param n_clusters: int, the number of clusters to create
    :return: Tuple[BaseEstimator, np.ndarray], the KMeans fitted clustering model and the cluster labels for each document
    """
    # Create an instance of the KMeans class from scikit-learn
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Fit the KMeans model to the document-term matrix
    kmeans.fit(matrix)

    # Obtain the cluster labels for each document
    cluster_labels = kmeans.labels_

    # Return the KMeans model and the cluster labels as a tuple
    return kmeans, cluster_labels

