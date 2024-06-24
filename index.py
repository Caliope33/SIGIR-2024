# indexing.py

import pyterrier as pt
import ir_datasets
import pandas as pd
import os

def document_generator(dataset):
    for doc in dataset.docs_iter():
        yield {
            'docno': doc.doc_id,
            'text': doc.text
        }

def create_index(dataset, index_path='./msmarco_index'):
    pt.init()  # Initialize PyTerrier
    
    indexer = pt.IterDictIndexer(index_path, meta=['docno', 'text'], meta_lengths=[20, 4096])
    index_ref = indexer.index(document_generator(dataset))
    return index_ref, index_path

def preprocess_query(query):
    import re
    # Remove problematic characters
    query = re.sub(r"[\"']", "", query)
    # Replace any other problematic characters with a space
    query = re.sub(r"[^\w\s]", " ", query)
    # Trim and ensure proper format
    query = query.strip()
    return query

def save_queries(dataset, queries_path='./queries.csv'):
    queries_list = []
    for query in dataset.queries_iter():
        preprocessed_query = preprocess_query(query.text)
        queries_list.append({'qid': query.query_id, 'query': preprocessed_query})

    queries_df = pd.DataFrame(queries_list)
    queries_df.to_csv(queries_path, index=False)

if __name__ == "__main__":
    dataset = ir_datasets.load("vaswani")
    index_ref, index_path = create_index(dataset)
    
    # Save the index reference path to a file
    index_ref_path = './index_ref.txt'
    with open(index_ref_path, 'w') as f:
        f.write(index_path)
    
    # Save the preprocessed queries
    queries_path = './queries.csv'
    save_queries(dataset, queries_path)
    
    print(f"Index created and saved at: {index_ref}")
    print(f"Index reference saved at: {index_ref_path}")
    print(f"Queries saved at: {queries_path}")

