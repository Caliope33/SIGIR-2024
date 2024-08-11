import os
import argparse
import numpy as np
import pandas as pd
import pytrec_eval
from tira.third_party_integrations import ir_datasets
import cherche.retrieve as retrieve
import cherche.rank as rank
from sentence_transformers import SentenceTransformer


def main(dataset_path):
    # Load dataset
    dataset = ir_datasets.load(dataset_path)

    docs = list(dataset.docs_iter())
    queries = list(dataset.queries_iter())

    # Print a few documents and queries to verify
    for doc in docs[:1]:
        print(f"Document ID: {doc.doc_id}")
        print(f"Document Text: {doc.text}")
        print("-----------")

    for query in queries[:1]:
        print(f"Query ID: {query.query_id}")
        print(f"Query Text: {query.text}")
        print("-----------")

    # Prepare relevance evaluations
    qrel = {
        k: {kk: int(vv) for kk, vv in v[["doc_id", "relevance"]].values}
        for k, v in pd.DataFrame(dataset.qrels_iter()).groupby("query_id")[
            ["doc_id", "relevance"]
        ]
    }

    documents = [{"id": doc.doc_id, "text": doc.text} for doc in dataset.docs_iter()]

    # Initial retrieval with Lunr
    retriever = retrieve.Lunr(key="id", on=["text"], documents=documents, k=100)

    # Run initial retrieval
    run = {
        query.query_id: {x["id"]: float(x["similarity"]) for x in retriever(query.text)}
        for query in dataset.queries_iter()
    }

    # Write initial retrieval results to file
    def write_trec_run(run, filename, run_name="run_name"):
        with open(filename, "w") as f:
            for query_id, doc_scores in run.items():
                for rank, (doc_id, score) in enumerate(
                    sorted(doc_scores.items(), key=lambda x: x[1], reverse=True),
                    start=1,
                ):
                    f.write(f"{query_id} Q0 {doc_id} {rank} {score} {run_name}\n")

    # Re-rank using neural model
    ranker = rank.Encoder(
        key="id",
        on=["text"],
        encoder=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode,
        k=50,
    )

    search = retriever + ranker
    search.add(documents=documents)

    # Run re-ranking
    run = {
        query.query_id: {x["id"]: float(x["similarity"]) for x in search(query.text)}
        for query in dataset.queries_iter()
    }

    # Write re-ranking results to file
    write_trec_run(run, "run.txt", run_name="Lunr-miniLML6")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run retrieval and ranking on a dataset."
    )
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    args = parser.parse_args()

    main(args.dataset_path)
