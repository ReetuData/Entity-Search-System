## Import libraries
import os;                             #https://github.com/numpy/numpy/issues/14868
os.environ["OMP_NUM_THREADS"] = "1"  
import sys
from pandas import DataFrame

# Parent Path Appended
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from data_preprocessing import read_dataframe

# Load preprocessed_df 
preprocessed_df = read_dataframe()

# define function to Tversky Index

import pandas as pd

def tversky_index_for_sets(df:pd.DataFrame, column_name:str, alpha=0.5, beta=0.5):
    """
    Calculate the Tversky Index between two sets created from a df column.

    Parameters:
    - df (pd.DataFrame): The df containing the data.
    - column_name (str): Column name to be converted into sets and compared.
    - alpha (float, optional): The weight assigned to common elements (intersection). Default is 0.5.
    - beta (float, optional): The weight assigned to distinct elements (set differences). Default is 0.5.

    Returns:
    - float: The Tversky Index, a value between 0 and 1, indicating the similarity between the two sets.
    
    Output: Tversky Index: 0.5

    Note:
    - This function converts the specified column into sets and calculates the Tversky Index between them.
    """
    # Convert the column values to sets
    sets = Preprocessed_df[column_name].str.split(',').apply(set)

    if len(sets) < 2:
        return "Insufficient data for comparison."

    set1 = sets.iloc[0]
    set2 = sets.iloc[1]

    common_elements = set1.intersection(set2)
    distinct_elements1 = set1.difference(set2)
    distinct_elements2 = set2.difference(set1)

    tversky_score = len(common_elements) / (len(common_elements) + alpha * len(distinct_elements1) + beta * len(distinct_elements2))
    return tversky_score

# Getting Tverkey Index 
data = {'ID': [1, 2, 3], 'Items': ['amazon com inc, microsoft corp, alphabet inc', 'microsoft corp, alphabet inc,microsoft corp, alphabet inc ', 'amazon com inc, microsoft corp,  nvidia corp']}
df = pd.DataFrame(preprocessed_df)

tversky_score = tversky_index_for_sets(preprocessed_df, 'title', alpha=0.7, beta=0.3)
print("Tversky Index:", tversky_score)
