## Import libraries
import os;                             #https://github.com/numpy/numpy/issues/14868
os.environ["OMP_NUM_THREADS"] = "1"  
import sys
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Parent Path Appended
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from data_preprocessing import read_dataframe

# Load preprocessed_df 
preprocessed_df = read_dataframe()

## define a function ot calculate cosine similarity using TF-IDF with Cosine Similarity
def calculate_cosine_similarity(df:DataFrame, text_column:str):
    
    """
    Compute the cosine similarity matrix for a text column of the df.

    Parameters:
    df (pd.DataFrame): The df containing the text data.
    text_column (str): The column containing the text data.

    Returns:
    pd.DataFrame: A df with the cosine similarity matrix.

    """
    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    
    # Fit and transform the text data to obtain TF-IDF features
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column])
    
    # Compute the cosine similarity between documents
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Create a df for the similarity scores
    similarity_df = pd.DataFrame(cosine_sim, columns=preprocessed_df.index, index=preprocessed_df.index)
    
    # returns df with similarity scores
    return similarity_df

# Getting similarity score from preprocessed data
similarity_matrix = calculate_cosine_similarity(preprocessed_df, 'title')
print(similarity_matrix)
