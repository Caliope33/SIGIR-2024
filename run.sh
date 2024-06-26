#!/bin/sh

python3 /app/indexing.py "$inputDataset" "$outputDir"
python3 /app/retrieve.py 
python3 /app/re-rank.py 

