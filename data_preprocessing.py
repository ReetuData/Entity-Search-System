## Data Preprocessing
from data_helpers import read_dataframe, convert_df_to_lowercase, handle_spec_char_and_diacritics, save_df_as_parquet
from data_loader import parent_path

# 1. Load dataframe from data_loader saved as parquet file
comp_ticker_df = read_dataframe()

# load DataFrame
print(comp_ticker_df.head(10))

# Convert the entire df to lowercase
converted_comp_ticker_df = convert_df_to_lowercase(comp_ticker_df)

print("\nDataFrame with all text data in lowercase:")
print(converted_comp_ticker_df)

# Handle special characters and diacritics in the entire DataFrame
preprocessed_df = handle_spec_char_and_diacritics(converted_comp_ticker_df)

# Print the modified DataFrame
print(preprocessed_df)

# Specifying the folder path and file name
file_name = "preprocessed_df.parquet"

# Save the df as a Pickle file in the folder
save_df_as_parquet(preprocessed_df, parent_path, file_name)
