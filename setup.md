
Setup python visual studio environment [with typing](https://code.visualstudio.com/docs/python/tutorial-fastapi):

Open the Settings editor (⌘,).
Search for "python type checking mode" and set it to basic for basic type checking. Pylance will now show diagnostics and warnings to catch simple type-related errors. Alternatively, you can set it to strict to enforce more advanced type checking rules.
Next, search for "Python inlay type hints", and enable inlay hints for "Variable Types" and "Function Return Types"

# Install environment
```
conda create -n rag python
conda activate rag
conda install ninja
conda install pytorch -c pytorch
conda install -c pytorch faiss-cpu=1.8.0
#conda install -c pytorch -c nvidia faiss-gpu=1.8.0

pip install colbert-ai srsly requests
#pip install ragatouille pyvespa requests
# upgrade to latest llama-index for SemanticSplitter
pip install llama-index --upgrade
#pip install llama-index-readers-web

# for docx import
pip install docx2txt
#https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface.html
pip install llama-index-embeddings-huggingface
#pip install llama-index-embeddings-instructor
#pip install transformers optimum[exporters]

pip install aiosqlite
pip install fastapi
pip install "uvicorn[standard]"

conda env export > environment.yml

conda deactivate 
conda remove --name rag --all
```

In visual studio code select conda environment 
⇧⌘P python: s


# Vespa

```
brew install vespa-cli

vespa config set target local
vespa clone colbert vespa-colbert && cd vespa-colbert

docker run --detach --name vespa --hostname vespa-container \
  --publish 8081:8080 --publish 19071:19071 \
  vespaengine/vespa

docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa



vespa deploy --wait 300

vespa document ext/1.json
vespa document ext/2.json
vespa document ext/3.json

vespa query 'yql=select * from doc where userQuery() or ({targetHits: 100}nearestNeighbor(embedding, q))'\
 'input.query(q)=embed(e5, "query: space contains many suns")' \
 'input.query(qt)=embed(colbert, @query)' \
 'query=space contains many suns'

vespa visit

docker rm -f vespa
```

# Run

Run FastAPI server

in dev
```
uvicorn main:app --reload
```

in prod
```
uvicorn main:app
```
