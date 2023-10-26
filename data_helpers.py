"""
Title: data_helpers.py
Author: Reetu Sharma PhD
Date: October 22, 2023

Description:

This script aims to create a few helper function utilities for

1. Preprocessing Data
2. Loading Transformed Data
...

"""

from pandas import read_parquet, DataFrame
from data_loader import parent_path
from string import punctuation
from os.path import join
from unidecode import unidecode

# 1. Read Data
def read_dataframe():
    return read_parquet(parent_path)

# 2. Function to convert entire DataFrame in lower case
def convert_df_to_lowercase(df: DataFrame):
    """
    Convert the entire df to lowercase

    Parameters:
        df (DataFrame): The df to be modified.

    Returns:
        df: The modified df.
    """
    data_frame = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    return data_frame

# 3. Function to handle punctuation and special characters
def handle_spec_char_and_diacritics(df: DataFrame):
    """
    Handle special characters and diacritics in text columns of df.

    Parameters:
        df (DataFrame): The df to be modified.

    Returns:
        DataFrame: The modified df.
    """
    for column_name in df.columns:
        if df[column_name].dtype == 'object':
            
            # Define a translation table to remove diacritics and standardize special characters
            translator = str.maketrans('', '', punctuation)

            # Apply the translation to each cell in the column
            
            """
            In this method, the lambda function processes each cell value, denoted as 'x,' by first 
            checking if it is a string. If it is indeed a string, the function applies the 'unidecode' 
            function to convert diacritics to their English letter equivalents. Subsequently, the 'translate' 
            method is employed to remove specific characters as specified in the 'translator' table, which was
            initially created to eliminate punctuation marks.
            
            """
            df[column_name] = df[column_name].apply(lambda x: unidecode(x).translate(translator) if isinstance(x, str) else x)
    
    return df

# Function to save preprocessed df
def save_df_as_parquet(df: DataFrame, folder_path:str, file_name:str):
    """
    Save a df as a Parquet file 

    Parameters:
        df (DataFrame): The df to be saved.
        folder_path (str): folder path where the Pickle file to be saved.
        file_name (str): The name of the Pickle file.

    Returns:
        None
    """
    # Create the full file path (folder path + file name)
    file_path = join(folder_path, file_name)

    # Save the df as pickle to the folder
    df.to_parquet(file_path)

    print(f"df saved as parquet to '{file_path}'")

    return df
