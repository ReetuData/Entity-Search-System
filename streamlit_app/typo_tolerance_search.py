# 1. import libraries
from pandas import DataFrame
from rank_bm25 import BM25Okapi
from Levenshtein import distance  

# 2. define a function to apply bm25 search

def apply_bm25_search(dataframe: DataFrame, text_column: str, query: str):
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

# 3. Define a function for typo tolerance search
def apply_typo_tolerant_search(dataframe: DataFrame, text_column: str, query: str, max_edit_distance: int):
    """
    Apply typo-tolerant search using Levenshtein distance to a DataFrame column.

    Args:
        dataframe (DataFrame): The DataFrame containing the text data.
        text_column (str): Text column.
        query (str): The search query.
        max_edit_distance (int): Maximum allowed edit distance for a match.

    Returns:
        DataFrame: The DataFrame with added typo-tolerant search scores.
    """
    # Calculate Levenshtein distances for the query against the DataFrame column
    distances = dataframe[text_column].apply(lambda text: distance(query, text))

    # Filter rows based on the maximum allowed edit distance
    typo_tolerant_df = dataframe[distances <= max_edit_distance]

    return typo_tolerant_df












