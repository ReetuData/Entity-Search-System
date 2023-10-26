## Import libraries
import os;                             #https://github.com/numpy/numpy/issues/14868
os.environ["OMP_NUM_THREADS"] = "1"  
import sys
from fuzzywuzzy import fuzz

# Parent Path Appended
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from data_preprocessing import read_dataframe

# Load preprocessed_df 
preprocessed_df = read_dataframe()

# define function to calculate similarity using Monge_Elken_method
def calculate_string_similarity(s1:str, s2:str):
    """
    Calculate string similarity between two input strings.
    
    Parameters:
                s1 (str): First input string.
                s2 (str): Second input string.
    
    Returns:
    int: Similarity score of input string as a percentage (0 to 100)
    """
    
    # Calculate the similarity score by the ratio function from fuzzywuzzy
    similarity_score = fuzz.ratio(s1, s2)
    return similarity_score

# Getting the similarity score from preprocessed df
string1 = "irsa investments  representations inc"
string2 = " industrial tech acquisitions ii inc"
similarity_score = calculate_string_similarity(string1, string2)
print("String Similarity:", similarity_score)


