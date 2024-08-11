this file countaines a pipeline of neural retrieval. using 'cherche' , a tool that makes implementing information retrieval easy and quick for begginers in this field.
a first stage retrieval is done using the sparse retrieval Lunr to get the top 100 candidates , then using an encoder "miniLM-L6" , we re-rank the candidates to return the top 50"
