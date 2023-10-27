# [Entity-Search-System](http://209.182.236.218:8501/)

![Entity-search-sysytem-app](ESI.png)

## Table of Contents

- [Overview](#overview)
- [Data Retrieval and Preprocessing](#data-retrieval-and-preprocessing)
- [Text Analysis and Search](#text-analysis-and-search)
- [Running the Application](#running-the-application)
- [Docker Configuration](#docker-configuration)
- [Installation](#Installation)
- [Use Cases](#Use-Cases)

### Overview

The objective of this project is to develop an **Entity Search System**, a key component of the data analytics toolkit. This system is designed to take a user's input, typically a keyword or a company name, and return a list of entities that closely match or are relevant to the input.

#### Project Scope

- The Entity Search System is a critical tool that empowers users to efficiently explore and find relevant entities within the data ecosystem. 
- It serves as the bridge between users and diverse datasets, enabling users to search for entities of interest without the need for exact matches.

#### Key Objectives

The primary objectives of our Entity Search System are as follows:

- **Efficient Matching**: To build a system that can efficiently match user input with entities in datasets and return entities that are most likely relevant to the user's query.

- **Flexibility**: To make the system flexible enough to handle various user scenarios.

- **User-Friendliness**: The system should be user-friendly and provide relevant suggestions or matches, even in cases where the user's query might not be precise.

## Data Retrieval and Preprocessing

### Initial Data Retrieval

The initial part of the script retrieves data from a JSON file hosted at the given [URL](#"https://www.sec.gov/files/company_tickers.json") and then performs data preprocessing steps.

- The "data_helpers.py" script provides essential functions for data retrieval and preprocessing.
  
- The `retrieve_data` function fetches data from a given URL, checks for an HTTP request and JSON parsing errors, and returns the data as a JSON representation.
  
- The data preprocessing starts with the loading of a Parquet file using `read_dataframe`.
  
- The data is then converted to lowercase using `convert_df_to_lowercase`.
  
- The special characters and diacritics are handled using `handle_spec_char_and_diacritics`.
  
- Finally, the preprocessed data is saved as a Parquet file with the name `preprocessed_df` and folder path using `save_df_as_parquet`.

        **Input**: URL (string): URL to perform an HTTP GET request on.

        **Output**: data_json (dict): JSON representation of the data.

## Text Analysis and Search

In a recent project, five methods were used for text analysis: BM25, Word2Vec, TF-IDF, Tversky Index, and Monge-Eiken Method.

### Methods for Text Analysis

1. **BM25 (Best Matching 25)**:
   - BM25 is a ranking function used for information retrieval.
   - BM25 calculates a relevance score for a document based on the frequency of query terms within the document.
   - It also considers the document's length and a tunable parameter to control term saturation. 
   - The BM25 formula combines these factors to score and rank documents for a given query.

   **Pros**:
   - **Effective Ranking**: BM25 provides highly effective document ranking for information retrieval tasks.
   - **Tunable Parameters**: It allows fine-tuning of parameters, making it adaptable to different datasets.
   - **Handles Long Documents**: BM25 performs well with longer documents, where simple methods like TF-IDF may fall short.

   **Cons**:
   - **Complexity**: The tuning of parameters can be complex and requires domain expertise.
   - **Not Suitable for Short Documents**: BM25 might not perform optimally on very short documents.
   - **Computationally Intensive**: Calculating BM25 scores for a large document collection can be computationally expensive.

   **When to Use**:
   - Use BM25 for information retrieval tasks where we need to rank and retrieve documents based on keyword relevance to a query. BM25 is well-suited for search engines, recommendation systems, and text summarization. It performs effectively on larger documents or when we have a corpus with varying document lengths.

2. **Word2Vec**:
   - **Description**: Word2Vec is a technique used for natural language processing and word embedding. 
   - Word2Vec uses a neural network to learn word representations from large text corpora. 
   - It learns to predict the context (surrounding words) of a target word, and in the process, it generates dense vector representations for words. 
   - These word vectors capture semantic relationships, enabling similarity measurements between words.

   **Pros**:
   - **Semantic Understanding**: Word2Vec captures semantic relationships and word similarities effectively.
   - **Word Embeddings**: It generates dense vector representations for words, which can be used for various NLP tasks.
   - **Transfer Learning**: Pre-trained Word2Vec models can be fine-tuned for specific applications with limited data.

   **Cons**:
   - **Large Memory Footprint**: Storing pre-trained Word2Vec models can require significant memory.
   - **Training Data Dependency**: Creating custom Word2Vec models may need substantial amounts of text data for training.
   - **Lack of Document Context**: Word2Vec focuses on word-level embeddings and does not directly capture document-level semantics.

   **When to Use**:
   - Use Word2Vec when you want to capture semantic relationships between words and obtain word embeddings. It's valuable for applications requiring word similarity, text classification, and natural language understanding. Pre-trained Word2Vec models are useful for transfer learning in tasks with limited training data.

3. **TF-IDF (Term Frequency-Inverse Document Frequency)**:
   - **Description**: TF-IDF is a classic method for information retrieval and text analysis. 
   - For a given term in a document, TF (Term Frequency) calculates how often the term appears. 
   - IDF (Inverse Document Frequency) measures the importance of the term by considering how many documents contain the term across the corpus. 
   - TF-IDF is the product of these two values and is used to rank and retrieve documents based on the relevance of terms.

   **Pros**:
   - **Simple and Effective**: TF-IDF is a straightforward and effective method for information retrieval.
   - **Interpretable**: The TF-IDF score for a term in a document is interpretable and easy to understand.
   - **Scalability**: TF-IDF works well with both small and large document collections.

   **Cons**:
   - **Limited Semantics**: It does not capture word semantics, making it less suitable for tasks that require understanding context.
   - **Lack of Term Relevance**: TF-IDF does not consider the relevance of terms within the document itself.
   - **No Term Ordering**: It treats terms as independent, without considering term ordering within a document.

   **When to Use**:
   - TF-IDF is a suitable choice when simplicity and interpretability are important. Use it for basic document retrieval, keyword extraction, and text summarization. 
   - It works well for tasks involving small to medium-sized document collections.

4. **Tversky Index**:
   - **Description**: The Tversky index is a similarity measurement that compares the intersection and difference between two sets. 
   -  The Tversky index takes two sets as input and computes a similarity score based on a set of parameters (alpha and beta) that control the balance between emphasizing common elements or differences between sets.
   -   
   -  The formula involves both intersection and union operations and is used for comparing documents, tags, or any set-based data.

  -  *Alpha (α)*: Alpha controls the emphasis on common elements between the two sets. When α is set to 1, the Tversky index focuses entirely on the intersection/shared elements of the sets. it makes it sensitive to the similarity of the sets based on their common elements.

   - *Beta (β)*: Beta controls the emphasis on the differences between the two sets. When β is set to 1, the Tversky index concentrates entirely on the complement/unique of the intersection. This makes the method sensitive to the differences between the sets.

   - **Tversky Index Formula**:
         The Tversky index formula is as follows:


         T(A, B) = (|A ∩ B|) / (|A ∩ B| + α|A \ B| + β|B \ A|)

         |A ∩ B| represents the size of the intersection of sets A and B.
         |A \ B| represents the size of the elements that are in set A but not in set B.
         |B \ A| represents the size of the elements that are in set B but not in set A.

         When α > β, the Tversky index will favor common elements, sensitive to similarities.
         When α < β, the Tversky index will favor differences, sensitive to dissimilarities.
         When α = β, the Tversky index becomes the Jaccard index, a balanced measure of similarity 
         based on the intersection and union of sets.

   **Pros**:
   - **Flexible Similarity Measurement**: The Tversky Index is highly flexible, allowing adjustments of alpha and beta parameters to customize the similarity measurement.
   - **Set-Based Comparisons**: It is well-suited for comparing sets of items, making it applicable to various tasks.

   **Cons**:
   - **Parameter Dependency**: The effectiveness of the Tversky Index depends on choosing appropriate values for the alpha and beta parameters.
   - **No Magnitude Information**: The index does not provide magnitude information about similarity.
   - **Complexity in Multi-Sets**: Handling multisets (sets with repeated elements) can be complex.

   **When to Use**:
   - Use the Tversky Index for set-based similarity comparisons in applications like information retrieval, recommendation systems, and data mining. 
   - It is versatile and can be customized with alpha and beta parameters to emphasize common elements or differences based on the task requirements. 
   - Effective for comparing sets of tags, documents, or items.

5. **Monge-Eiken Method**:
   - **Description**: The Monge-Eiken method is a technique used in text clustering and categorization. 
   -  This method operates by calculating the similarity between documents using a predefined set of features or attributes. 
   -  The similarity is calculated using a specific formula, often considering the frequency of shared features. 
   -  The documents are then clustered based on their similarity scores.

   **Pros**:
   - **Effective Clustering**: The Monge-Eiken method is known for its effectiveness in text clustering based on content similarity.
   - **Handles Diverse Data**: It can be applied to a wide range of text data types, including documents and textual data from different domains.
   - **Interpretable Features**: Users can define and interpret features used for similarity calculation.

   **Cons**:
   - **Feature Selection Challenge**: Choosing the right features for similarity measurement can be challenging and may require domain knowledge.
   - **Sensitivity to Feature Weights**: The method's performance can be sensitive to the weights assigned to different features.
   - **Scalability**: For very large datasets, the method may become computationally intensive.

   **When to Use**:
   - Choose the Monge-Eiken method when we need to cluster or categorize documents based on content similarity.
   -  It is suitable for text categorization, clustering, and topic modeling. 
   -  The method's flexibility allows it to be applied to various textual data types and domains.

In the context of text analysis, a meticulous selection process was undertaken, ultimately leading to the choice of five distinct models. Among these, three were singled out for the analysis—namely, BM25, Word2Vec, and TF-IDF. 

The decision to employ these specific methods was influenced by several factors, including the structured nature of the dataset, the availability of computational resources, and the specific analytical needs of the task at hand.

Following the execution of text analysis employing these three methods, the final determination of the optimal approach was made based on the evaluation metrics which encompassed significant measures such as F1 Score, Accuracy, Recall, and Precision. This rigorous approach ensured the selection of the most suitable and effective method for the given context.

### F1 Score:
The F1 Score is a harmonic mean of precision and recall. It's a good metric for binary classification tasks, where there is an imbalance between the positive and negative classes. In this case, "BM25" had the highest F1 Score (0.006302) among the three methods.

### Accuracy:
Accuracy is the ratio of correctly predicted instances to the total instances. It's a common metric for classification tasks. "BM25" also had the highest accuracy (0.003161).

### Recall:
Recall is the ability of the model to identify all relevant instances in the dataset. It's important when you want to minimize false negatives. "BM25" had the highest recall (0.003161).

### Precision:
Precision is the ability of the model to identify only the relevant instances among the predicted positive instances. In this case, all methods had precision equal to 1.0, so it didn't contribute to the decision.

Given that "BM25" performed the best in terms of F1 Score, Accuracy, and Recall, it was selected as the best method for the given dataset.

## Search Algorithm using BM25 Search

- The BM25 Search script is a tool that enhances text retrieval within a data frame using the BM25 (Best Matching 25) search algorithm.
- BM25 is a ranking method designed to efficiently retrieve relevant documents based on keyword relevance to a query. 
- To use this tool, you need to import the `rank_bm25` library and pass the preprocessed DataFrame into the search function named `apply_bm25_search`.

- The `apply_bm25_search` function enables BM25 to search on a specified text column of a data frame. 
- The function tokenizes text, calculates BM25 scores, and ranks the DataFrame. 
- -To use it, provide the name of the text column within the preprocessed DataFrame and set a search query for the BM25 algorithm. 
- It returns a DataFrame enriched with BM25 scores, sorted by relevance. 
- We can review and interpret the results by printing the top-relevant records based on BM25 scores.

## Technical Explanation of Text Search and Typo-Tolerant Search Functions

**Tool 1: BM25 Search**

The `apply_bm25_search` function is a text retrieval method based on the BM25 algorithm, which is commonly used in information retrieval. Here's how it operates:

**Tokenization and Preprocessing**: In this step, text data within a DataFrame column is tokenized, which means it's divided into individual words or terms. 

**BM25 Instance Creation**: An instance of the BM25 algorithm is created using the tokenized documents. This instance essentially encodes the statistical properties of the text data.

**Query Tokenization**: The search query is also tokenized, which means it's broken down into its constituent words or terms.

**BM25 Score Calculation**: BM25 scores are calculated for each document in the DataFrame based on its relevance to the query. The algorithm takes into account factors like term frequency and document length normalization.

**Ranking Documents**: The data frame is sorted and ranked according to the calculated BM25 scores. This results in the most relevant documents appearing at the top.

**Returns**: The function returns the sorted DataFrame,
allowing users to quickly identify documents that are most relevant to the query.

**Tool 2: Typo-Tolerant Search**

- The `apply_typo_tolerant_search` function provides a mechanism for conducting the typo-tolerant search. 
- It leverages the Levenshtein distance, also known as edit distance, to find approximate matches.
-  Here's how it functions:

**Levenshtein Distance Calculation**:
- Levenshtein distance measures the number of single-character edits (insertions, deletions, or substitutions) required to transform one string into another. 
- In this context, it calculates the edit distance between the search query and each document in the DataFrame column.

**Filtering Based on Edit Distance**:
- Rows in the data frame are filtered based on the maximum allowed edit distance. 
- This threshold determines how similar the documents must be to the query.
-  If the edit distance falls within the specified limit, the document is considered a match.

**Returns**: The function returns a DataFrame containing documents that meet the typo-tolerant search criteria, making it possible to find approximate matches.

## Installation

### Docker 

        # Clone the repository
        git clone [https://github.com/ReetuData/Entity-Search-System](https://github.com/ReetuData/Entity-Search-System.git)

        # Navigate to the application root folder
        cd Entity-Search-System-App

        # Run the docker-compose YML file
        docker-compose up -d
    

### Without Docker

        Run the app using: # Clone the repository
        git clone [https://github.com/ReetuData/Entity-Search-System](https://github.com/ReetuData/Entity-Search-System.git)

        # Navigate to the application root folder
        cd Entity-Search-System-App

        # Install Python module requirements
        pip install -r requirements.txt

        # Run the Streamlit application
        streamlit run app.py
## Use Cases

These functions are valuable for a variety of applications:

**Information Retrieval**: The BM25 search function is ideal for quickly retrieving documents that are most relevant to a query, making it suitable for search engines, recommendation systems, and content retrieval.

**Approximate Matching**: The typo-tolerant search function is beneficial when approximate matches are acceptable, such as when dealing with user queries that may contain spelling errors.

