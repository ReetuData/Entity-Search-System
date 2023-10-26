# 1. import libraries
import pandas as pd
import gensim
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

import sys
import os

# Parent Path Appended
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from data_preprocessing import read_dataframe

# 2. Load dataframe from preprocessed_df 
preprocessed_df = read_dataframe()

# Create Word2Vec model
model = gensim.models.Word2Vec(preprocessed_df['title'], min_count=1, vector_size=100, sg=0)  # sg=0 for CBOW, sg=1 for Skip-gram

# 3. Define the calculate_similarity function using Word2Vec
def calculate_similarity_word2vec(model, query, documents):
    query_tokens = query.split()  # Split the query into tokens
    scores = []
    for doc in documents:
        doc_tokens = doc.split()  # Split each document into tokens
        similarity = model.wv.wmdistance(query_tokens, doc_tokens)
        scores.append(1 / (1 + similarity))
    return scores

# 4. Define the calculate_similarity function using TF-IDF
def calculate_similarity_tfidf(query, documents):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    query_vector = tfidf_vectorizer.transform([query])
    scores = (tfidf_matrix * query_vector.T).toarray().flatten()
    return scores

# 5. Define the calculate_similarity function using BM25
def calculate_similarity_bm25(query, documents):
    tokenized_documents = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_documents)
    scores = bm25.get_scores(query.split())
    return scores

# 6. Define a function to compare and evaluate the methods
def compare_methods(df, query):
    methods = ["TF-IDF", "Word2Vec", "BM25"]
    
    results = []  # Create an empty list to store the results
    
    for method in methods:
        if method == "TF-IDF":
            scores = calculate_similarity_tfidf(query, df['title'])
        elif method == "Word2Vec":
            scores = calculate_similarity_word2vec(model, query, df['title'])
        elif method == "BM25":
            scores = calculate_similarity_bm25(query, df['title'])
        
        # Ensure that binary_scores and true_labels have the same length
        binary_scores = [1 if score > 0.3 else 0 for score in scores]
        true_labels = [1] * len(scores)  # Use len(scores) as the number of samples
        
        f1 = f1_score(true_labels, binary_scores)
        accuracy = accuracy_score(true_labels, binary_scores)
        recall = recall_score(true_labels, binary_scores)
        precision = precision_score(true_labels, binary_scores, zero_division=1)
        
        results.append({
            'Method': method,
            'F1 Score': f1,
            'Accuracy': accuracy,
            'Recall': recall,
            'Precision': precision
        })

    # Create a DataFrame to store the results
    result_df = pd.DataFrame(results)
    
    return result_df

# Specify a query
query = "apple company"  # Change the query as needed

# 7. Call the compare_methods function with the DataFrame and query
result_df = compare_methods(preprocessed_df, query)

# Display the results
print(result_df)
