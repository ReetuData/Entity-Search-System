
## Import libraries
import sys
import os;                             #https://github.com/numpy/numpy/issues/14868
os.environ["OMP_NUM_THREADS"] = "1"  
from pandas import DataFrame
from rank_bm25 import BM25Okapi

# Parent Path Appended
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from data_preprocessing import read_dataframe

# Load preprocessed_df 
preprocessed_df = read_dataframe()


# define function to caluculate BM25 Index 
def index_with_bm25(df:DataFrame, text_column:str):
    
    """
    Create and return a BM25Okapi indexer for a test column of a df.

    Parameters:
    df (pd.DataFrame): The df with the text data.
    text_column (str): The column name of df that contains text data.

    Returns:
    BM25Okapi: A BM25Okapi indexer for the provided text data.
    """

    # Extract the text data from the text column
    corpus = df[text_column].tolist()

    # Tokenize the documents
    tokenized_corpus = [doc.split() for doc in corpus]

    # Create an instance of the BM25Okapi indexer
    bm25 = BM25Okapi(tokenized_corpus)

    # Return the BM25 indexer
    return bm25

# Index the 'text' column using BM25
bm25_indexer = index_with_bm25(preprocessed_df, 'title')

# Retrieve the top 5 documents similar to a query in Preprocessed_df
query = "irsa investments  representations inc"
n = 5
scores = bm25_indexer.get_scores(query.split())
top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
top_similar_documents = preprocessed_df.loc[top_indices]
print(top_similar_documents)
