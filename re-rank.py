# rerank.py

import pyterrier as pt
import pandas as pd
import os

def load_initial_run(initial_run_path='./initial_run.csv'):
    return pd.read_csv(initial_run_path)

def perform_reranking(index_ref, initial_run_df):
    pl2 = pt.BatchRetrieve(index_ref, wmodel="PL2")

    reranked_run = []
    grouped = initial_run_df.groupby('qid')
    for queryid, group in grouped:
        querytext = group.iloc[0]['query']
        results = pl2.search(querytext)
        for rank, res in enumerate(results.itertuples()):
            reranked_run.append({
                'qid': queryid,
                'docno': res.docno,
                'score': res.score,
                'rank': rank + 1
            })

    return pd.DataFrame(reranked_run)

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

    # Load the initial run results
    initial_run_path = './initial_run.csv'
    if not os.path.exists(initial_run_path):
        raise FileNotFoundError(f"Initial run file not found: {initial_run_path}")

    initial_run_df = load_initial_run(initial_run_path)
    print("Initial run loaded")

    # Perform re-ranking
    reranked_run_df = perform_reranking(index_ref, initial_run_df)
    print("Re-ranked run loaded")
    print(reranked_run_df)

    # Save the re-ranked results to a file
    reranked_run_df.to_csv('./reranked_results.csv', index=False)
    print("Re-ranked results saved to './reranked_results.csv'")

