import ir_datasets
import os

# Charger le dataset Vaswani
dataset = ir_datasets.load("vaswani")

# Créer le répertoire "my_dataset" s'il n'existe pas
os.makedirs('vaswani', exist_ok=True)

# Chemins des fichiers dans le répertoire "my_dataset"
docs_path = os.path.join('vaswani', 'vaswani_docs.txt')
queries_path = os.path.join('vaswani', 'vaswani_queries.txt')
qrels_path = os.path.join('vaswani', 'vaswani_qrels.txt')

# Créer des fichiers pour stocker les données
with open(docs_path, 'w', encoding='utf-8') as doc_file, \
     open(queries_path, 'w', encoding='utf-8') as query_file, \
     open(qrels_path, 'w', encoding='utf-8') as qrel_file:
    
    # Écrire les documents
    for doc in dataset.docs_iter():
        doc_file.write(f"{doc.doc_id}\t{doc.text}\n")
    
    # Écrire les requêtes
    for query in dataset.queries_iter():
        query_file.write(f"{query.query_id}\t{query.text}\n")
    
    # Écrire les jugements de pertinence
    for qrel in dataset.qrels_iter():
        qrel_file.write(f"{qrel.query_id}\t{qrel.doc_id}\t{qrel.relevance}\n")

print("Le dataset a été téléchargé et sauvegardé localement dans le répertoire 'my_dataset'.")

