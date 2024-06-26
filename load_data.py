import sys
import ir_datasets
import os

def save_dataset_locally(dataset_name):
    # Load the specified dataset
    dataset = ir_datasets.load(dataset_name)
    
    # Create a directory named after the dataset if it doesn't exist
    dataset_dir = dataset_name.lower().replace(' ', '_')  # Create a directory name from dataset name
    os.makedirs(dataset_dir, exist_ok=True)

    # Paths for the files in the dataset directory
    docs_path = os.path.join(dataset_dir, f'{dataset_name}_docs.txt')
    queries_path = os.path.join(dataset_dir, f'{dataset_name}_queries.txt')
    qrels_path = os.path.join(dataset_dir, f'{dataset_name}_qrels.txt')

    # Create files to store the data
    with open(docs_path, 'w', encoding='utf-8') as doc_file, \
         open(queries_path, 'w', encoding='utf-8') as query_file, \
         open(qrels_path, 'w', encoding='utf-8') as qrel_file:
        
        # Write documents to file
        for doc in dataset.docs_iter():
            doc_file.write(f"{doc.doc_id}\t{doc.text}\n")
        
        # Write queries to file
        for query in dataset.queries_iter():
            query_file.write(f"{query.query_id}\t{query.text}\n")
        
        # Write relevance judgments to file
        for qrel in dataset.qrels_iter():
            qrel_file.write(f"{qrel.query_id}\t{qrel.doc_id}\t{qrel.relevance}\n")
    
    print(f"The dataset '{dataset_name}' has been downloaded and saved locally in the directory '{dataset_dir}'.")

# Check if the script is being run directly
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the dataset name as an argument.")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    save_dataset_locally(dataset_name)

