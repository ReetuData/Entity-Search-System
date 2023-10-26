# 1. import libraries
from pandas import DataFrame
from rank_bm25 import BM25Okapi
import sys
import os

# Parent Path Appended
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from data_preprocessing import read_dataframe

# Load preprocessed_df 
preprocessed_df = read_dataframe()

# 2. Define function to apply bm25 search
def apply_bm25_search(dataframe: DataFrame, text_column:str, query:str):
    
    """
    Apply BM25 search algorithm to a DataFrame column.

    Args:
        dataframe (DataFrame): The DataFrame containing the text data.
        text_column (str): Text column
        query (str): The search query.

    Returns:
        DataFrame: The DataFrame with added BM25 scores.
    """
    # Tokenize and preprocess the text column
    dataframe['tokenized_text'] = dataframe[text_column].apply(lambda text: text.split())

    # Create a BM25 instance
    tokenized_documents = dataframe['tokenized_text'].tolist()
    bm25 = BM25Okapi(tokenized_documents)

    # Tokenize the query
    tokenized_query = query.split()

    # Calculate BM25 scores for the query against the DataFrame column
    scores = bm25.get_scores(tokenized_query)

    # Add the BM25 scores to the DataFrame
    dataframe['bm25_score'] = scores

    # Sort and rank the DataFrame rows based on BM25 scores
    dataframe = dataframe.sort_values(by='bm25_score', ascending=False)

    return dataframe

# 3. Specify column name from preprocessed_df
# specify a query
query = "amazon com inc"

# Apply BM25 search to the DataFrame
BM25_result_df = apply_bm25_search(preprocessed_df, 'title', query)

# Display the sorted DataFrame
print(BM25_result_df.head(10))

