The project uses pretrained transformer embeddings, to embed the docs. when a query is provided the query gets converted to vector which is embedding using the same transformer used in the documents.
It than uses cosine similarity to retrieve relevant doc from the docs.
The embeddings are stored in .vec and its meta are in .meta file.
