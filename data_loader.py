# import libraires 
from pandas import json_normalize
from requests import get, exceptions
from os import getcwd

# Function to retrieve a response
def retrieve_data(url: str):
    """
    Inputs
        - url (string): URL to perform a HTTP get request on
    
    Output
        - data_json (dict or list): JSON representation
    """

    # Send a GET request to the URL
    response = get(url)

    # Try Except blocks
    try:
        # Send a GET request to the URL
        response = get(url)

        # Check if the status code is 200
        response.raise_for_status()  # Raise an exception for non-200 status codes

        # Parse the JSON data from the response content
        data_json = response.json()
        
    except exceptions.RequestException as e:
        
        # Handle exceptions related to the request
        print(f"HTTP Request Error: {e}")
        data_json = None
    
    except ValueError as e:
        # Handle exceptions related to JSON parsing
        print(f"JSON Parsing Error: {e}")
        data_json = None

    return data_json

# URL of the JSON file
url = "https://www.sec.gov/files/company_tickers.json"

# Get the data_json
data_json = retrieve_data(url)

comp_ticker_values = list(data_json.values())

# DataFrame from a normalization
comp_ticker_df = json_normalize(comp_ticker_values)

# Get first 10 rows
print(comp_ticker_df.head(10))

# Saving data to load in further preprocessing in other Python scripts

## Specify the full path to save the parquet file
parent_path = getcwd() + '/src/mini_projects/mp5/datasets'

## Save the DataFrame to the parquet file with snappy compression
comp_ticker_df.to_parquet(f"{parent_path}/comp_ticker_df.parquet", compression='snappy')




