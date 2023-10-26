# import libraries
from pandas import DataFrame
from rank_bm25 import BM25Okapi
from gensim.models import Word2Vec

import sys
import os

# Parent Path Appended
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from data_preprocessing import read_dataframe
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


# 2. Load dataframe from preprocessed_df 
preprocessed_df = read_dataframe()

# 3. Define the calculate_similarity function
def calculate_similarity(method, query, documents):
    if method == "TF-IDF":
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        query_vector = tfidf_vectorizer.transform([query])
        scores = (tfidf_matrix * query_vector.T).toarray().flatten()
        
    elif method == "Word2Vec":
        model = Word2Vec(documents, min_count=1, vector_size=100)
        query_tokens = query.split()
        document_tokens = [doc.split() for doc in documents]
        scores = []
        for doc_tokens in document_tokens:
            similarity = model.wv.wmdistance(query_tokens, doc_tokens)
            scores.append(1 / (1 + similarity))
            
    elif method == "BM25":
        tokenized_documents = [doc.split() for doc in documents]
        bm25 = BM25Okapi(tokenized_documents)
        scores = bm25.get_scores(query.split())
        
    else:
        raise ValueError("Invalid method. Supported methods are: 'TF-IDF', 'Word2Vec', 'BM25'")
    return scores

# 4. Define a function to compare and evaluate the methods
def compare_methods(df, query, methods):
    results = []  # Create an empty list to store the results
    
    for method in methods:
        scores = calculate_similarity(method, query, df['title'].tolist())
        
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
    result_df = DataFrame(results)
    
    return result_df

# Specify the methods you want to compare
methods_to_compare = ["TF-IDF", "Word2Vec", "BM25"]

# Specify a query
query = "apple company"  # Change the query as needed

# 5. Call the compare_methods function with the DataFrame, query, and methods
result_df = compare_methods(preprocessed_df, query, methods_to_compare)

# Display the results
print(result_df)
