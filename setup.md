
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


Known issues:

When shutting down the server we get this
````
miniconda/base/envs/rag/lib/python3.12/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
```

caused by use of tqdm.tqdm() in

colbert/search/index_loader.py -> _load_doclens
colbert/indexing/codecs/residual_embeddings.py -> load_chunks

might be related to this:
https://github.com/tqdm/tqdm/issues/1321
