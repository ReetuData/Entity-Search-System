## Import libraries

import os;                             # https://github.com/numpy/numpy/issues/14868
os.environ["OMP_NUM_THREADS"] = "1"  
import nltk
nltk.data.path.append('/home/rsharma/Reetu_Test/Pathrise_DS_Reetu_Tutorials/src/mini_projects/mp5/NLTK dependencies')
# Download the 'punkt' data
nltk.download('punkt')
import string
import pickle
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize 

## Load preprocessed df saved as pkl file

def load_df_from_pickle(file_path:str):
    """
    Load a df from a Pickle file.

    Parameters:
    - file_path (str): Pickle file path.

    Returns:
    - pd.DataFrame: The loaded df.
    """
    try:
        with open(file_path, 'rb') as file:
            loaded_df = pickle.load(file)
        return loaded_df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: An error occurred while loading the Pickle file. {str(e)}")
        return None

# Retrieving pickle file
file_path = 'src/mini_projects/mp5/Datasets/Preprocessed_df.pkl'
Preprocessed_df = load_df_from_pickle(file_path)
 

def train_word2vec_embeddings(df:pd.DataFrame, column_name:str, vector_size=100, window=5, min_count=1, sg=0, epochs=10):
    """
    Train Word2Vec embeddings on text data in a df column.

    Parameters:
    - df (pd.DataFrame): The df with the text data.
    - column_name (str): TColumn name with text data.
    - vector_size (int, optional): The dimensionality of the word vectors. Default is 100.
    - window (int, optional): The maximum distance between the current and predicted word within a sentence. Default is 5.
    - min_count (int, optional): Ignores all words with a total frequency lower than this. Default is 1.
    - sg (int, optional): Training algorithm: 0 for CBOW, 1 for Skip-gram. Default is 0 (CBOW).
    - epochs (int, optional): Number of training epochs. Default is 10.

    Returns:
    - gensim.models.word2vec.Word2Vec: The trained Word2Vec model.
    
    Note:
    - This function uses Gensim to train Word2Vec embeddings on the text data in the specified column.
    """
    # Tokenize the text data
    df[column_name] = df[column_name].apply(lambda x: word_tokenize(x))

    # Train Word2Vec model
    model = Word2Vec(
        df[column_name],
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs
    )

    return model

word2vec_model = train_word2vec_embeddings(Preprocessed_df, 'title', vector_size=100, window=5, min_count=1, sg=0, epochs=10)
word_vector = word2vec_model.wv['microsoft']
print("Word Vector for 'microsoft':", word_vector)










