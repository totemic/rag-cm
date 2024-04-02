
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
conda install -c conda-forge pybind11

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

conda env export --no-build > environment.yml

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

# Build docker
ECR_REPOSITORY_NAME=rag-cm
ECR_AWS_REGION=us-west-2

VERSION=$( cat VERSION )
echo "export BUILD_VERSION=${VERSION}" >> $BASH_ENV
echo 'export FULL_REPOSITORY_NAME="${AWS_ACCOUNT_ID}.dkr.ecr.${ECR_AWS_REGION}.amazonaws.com/${ECR_REPOSITORY_NAME}"' >> $BASH_ENV
echo 'export IMAGE_TAGGED_VERSION_BUILD="${FULL_REPOSITORY_NAME}:${BUILD_VERSION}-${CIRCLE_BUILD_NUM}"' >> $BASH_ENV
echo 'export IMAGE_TAGGED_VERSION="${FULL_REPOSITORY_NAME}:${BUILD_VERSION}"' >> $BASH_ENV
echo 'export IMAGE_TAGGED_LATEST="${FULL_REPOSITORY_NAME}:latest"' >> $BASH_ENV


docker build -t $IMAGE_TAGGED_VERSION_BUILD -t $IMAGE_TAGGED_VERSION -t $IMAGE_TAGGED_LATEST --build-arg IMG_VERSION=$BUILD_VERSION --build-arg IMG_BUILD_NUMBER=$CIRCLE_BUILD_NUM .


# Known issues that need to be fixed in the ColBERT dependency

## Semaphore leak
When shutting down the server we get this
```
miniconda/base/envs/rag/lib/python3.12/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
```

caused by use of tqdm.tqdm() in

colbert/search/index_loader.py -> _load_doclens
colbert/indexing/codecs/residual_embeddings.py -> load_chunks

might be related to this:
https://github.com/tqdm/tqdm/issues/1321

## ColBERT creating Checkpoint objects in multiple places
ColBERT creates a `Checkpoint` object in `Searcher`, `Indexer` and parent class `CollectionIndexer`, `IndexUpdater`

## ColBERT not keeping different parts in sync
When we have a searcher loaded and rebuild the index through `Indexer`, the searcher does not know anything about the added components. We currently have to trigger a reload of the searcher to make this work but it would be better if we could avoid this.
Details:
When indexing, `Indexer` through `CollectionIndexer` calculated `doclens`
`Searcher` stores its own `IndexScorer(IndexLoader)` in `self.ranker` that is not updated when the index is updated. Its `self.doclens` is only loaded at initialization. 

## No control over the passage ids inside ColBERT
ColBERT assigns passage ids in sequential order. We can't insert our own ids when building the index. And when adding elements, it calculates the next ids based on doclens. The problem is that when you delete elements from the index and the insert more elements to it, self.doclens wil return the amount of currently active passages (excluding the deleted ones). But the indexes in sql still have holes in them. When we now generate new ids following what ColBERT wants to use, then these ids are already used in the database.