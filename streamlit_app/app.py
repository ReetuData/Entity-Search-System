import streamlit as st
from typo_tolerance_search import apply_bm25_search, apply_typo_tolerant_search
from data_preprocessing import read_dataframe
from pandas import DataFrame

# Preprocessed DataFrame
def load_data():
    return read_dataframe()

# Helper Function with caching
def compute_search(frame: DataFrame, chosen_col: str, query: str, m_distance: int) -> DataFrame:
    # Compute result dataframe
    result_df = apply_bm25_search(frame, chosen_col, query)

    # Afterward, perform typo and tolerance search
    final_df = apply_typo_tolerant_search(result_df, chosen_col, query, m_distance)

    return final_df

st.markdown("<h1 style='text-align:center'> Entity Search System </h1>", unsafe_allow_html=True)

# Load DataFrame
df = load_data()

# Design
col_1, col_2, col_3 = st.columns(3, gap='medium')

# Context Manager
with col_1:
    col_selection = st.selectbox("Choose a column for searching: ", options=df.columns.tolist(), index=2)

with col_2:
    search_query = st.text_input("Type in an entity to search: ", value='Amazon Inc')

with col_3:
    edit_distance = st.number_input("Type in a distance for tuning: ", min_value=0, max_value=3, value=1)


result_type = st.selectbox("Choose a result type:", options=['BM25 Search', 'Typo Tolerance'])


# Search Button
if st.button("Search"):

    st.info(f"Calculating {result_type} for query: {search_query}.")
    result_df = apply_bm25_search(df, col_selection, search_query)
    
    if result_type == 'BM25 Search':
        # Apply computed search
        st.success(f"DataFrame loaded for query: {search_query}.")
        st.dataframe(result_df, use_container_width=True)
        

    elif result_type == 'Typo Tolerance':
        final_df = apply_typo_tolerant_search(result_df, col_selection, search_query, edit_distance)
        st.success(f"DataFrame loaded for query: {search_query}.")
        st.dataframe(final_df, use_container_width=True)
    
    st.balloons()






