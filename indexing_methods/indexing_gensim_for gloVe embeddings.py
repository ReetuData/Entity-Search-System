## Import libraries

import os;                             #https://github.com/numpy/numpy/issues/14868
os.environ["OMP_NUM_THREADS"] = "1"  
import gensim.downloader as api
from pandas import DataFrame
from rank_bm25 import BM25Okapi
import sys

# Parent Path Appended
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from data_preprocessing import read_dataframe

# Load preprocessed_df 
preprocessed_df = read_dataframe()


def calculate_similarity_with_glove(df:DataFrame, column_name:str, text1:str, text2:str, word_vectors=None):
    """
    Calculate text similarity using GloVe word embeddings with Gensim.

    Parameters:
    - df (DataFrame): The df with the text data.
    - column_name (str): Column name with text data.
    - text1 (str): The first text to compare.
    - text2 (str): The second text to compare.
    - word_vectors (gensim.models.keyedvectors.KeyedVectors, optional): Pre-trained GloVe word vectors.

    Returns:
    - float: Similarity score based on GloVe embeddings, where a higher score indicates greater similarity.
    
    Note:
    - This function uses pre-trained GloVe word vectors from Gensim to calculate text similarity.
    """
    if word_vectors is None:
        print("Please provide pre-trained GloVe word vectors.")
        return None

    if column_name not in df:
        print(f"'{column_name}' column not found in the df.")
        return None

    if text1 not in df[column_name].values or text2 not in df[column_name].values:
        print("One or both input texts not found in the df.")
        return None

    text1_embedding = softcossim([word_vectors[text1]], [word_vectors[word] for word in df[column_name]], word_vectors)
    text2_embedding = softcossim([word_vectors[text2]], [word_vectors[word] for word in df[column_name]], word_vectors)

     # Measure similarity with text1's embedding
    similarity_score = text1_embedding[0][0] 
    return similarity_score

# Load pre-trained GloVe word vectors
glove_vectors = api.load("glove-wiki-gigaword-100")  

similarity_score = calculate_similarity_with_glove(preprocessed_df, 'title', 'investments', 'representations', glove_vectors)
print("Similarity Score:", similarity_score)