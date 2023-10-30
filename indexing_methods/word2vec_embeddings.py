## Import libraries
import sys
import os;                             #https://github.com/numpy/numpy/issues/14868
os.environ["OMP_NUM_THREADS"] = "1"  
import nltk
import numpy as np
nltk.data.path.append('/....../mini_projects/mp5/NLTK dependencies')
# Download the 'punkt' data
nltk.download('punkt')
from pandas import DataFrame
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize 

# Parent Path Appended
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from data_preprocessing import read_dataframe

# Load preprocessed_df 
preprocessed_df = read_dataframe()

def train_word2vec_embeddings(df:pd.DataFrame, column_name:str, vector_size=100, window=5, min_count=1, sg=0, epochs=10):
    """
    Train Word2Vec embeddings on text data in a data frame column.

    Parameters:
    - df (pd.DataFrame): The df with the text data.
    - column_name (str): TColumn name with text data.
    - vector_size (int, optional): The dimensionality of the word vectors. The default is 100.
    - window (int, optional): The maximum distance between the current and predicted word within a sentence. The default is 5.
    - min_count (int, optional): Ignores all words with a total frequency lower than this. The default is 1.
    - sg (int, optional): Training algorithm: 0 for CBOW, 1 for Skip-gram. The default is 0 (CBOW).
    - epochs (int, optional): Number of training epochs. The default is 10.

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

word2vec_model = train_word2vec_embeddings(preprocessed_df, 'title', vector_size=100, window=5, min_count=1, sg=0, epochs=10)
word_vector = word2vec_model.wv['microsoft']
print("Word Vector for 'microsoft':", word_vector)










