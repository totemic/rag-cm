# rag-cm

RAG database that leverages [ColBERTv2](https://github.com/stanford-futuredata/ColBERT) and it's highly efficient PLAID index for retrieval. Includes document ingesting and chunking. Document storage uses sqlite3 to provide additional flexibility for retrieval, e.g. to retrieve passages around the retrieved chunks.