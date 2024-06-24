# retrieve.py

import pyterrier as pt
import pandas as pd
import re
import os
import pytrec_eval
import pandas as pd

def load_queries(queries_path='./queries.csv'):
    return pd.read_csv(queries_path)

def perform_retrieval(index_ref, queries_df):
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")

    run = {}
    for _, row in queries_df.iterrows():
        queryid = row['qid']
        querytext = row['query']
        results = bm25.search(querytext)
        run[queryid] = {res.docno: rank + 1 for rank, res in enumerate(results.itertuples())}

    return run

if __name__ == "__main__":
    pt.init()  # Initialize PyTerrier

    # Load the index path from the file
    index_ref_path = './index_ref.txt'
    if not os.path.exists(index_ref_path):
        raise FileNotFoundError(f"Index reference file not found: {index_ref_path}")

    with open(index_ref_path, 'r') as f:
        index_path = f.read().strip()

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index path not found: {index_path}")

    index_ref = pt.IndexRef.of(index_path)

    # Load the preprocessed queries
    queries_path = './queries.csv'
    queries_df = load_queries(queries_path)
    print(queries_df)

    run = perform_retrieval(index_ref, queries_df)
    print(run)


