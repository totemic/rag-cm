from typing import Any
import requests
import time
import sqlite3

from constants import (
    DB_FILE_PATH, 
    INDEX_NAME,
    INDEX_PATH,
    # INDEX_NAME_RAGA,
    # INDEX_PATH_RAGA
)
import db

from pathlib import Path
from colbert_util import ColBERT
# from ragatouillefork import RAGPretrainedModel


from llama_index.core import SimpleDirectoryReader
# from llama_index.readers.web import SimpleWebPageReader

from llama_index.core.schema import (
    BaseNode,
    Document,
)
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#from colbert.data.collection import Collection
from dbcollection import DbCollection



def open_sqlite() -> sqlite3.Connection:
    con: sqlite3.Connection = sqlite3.connect(DB_FILE_PATH)
    cursor: sqlite3.Cursor = con.cursor()

    # see https://charlesleifer.com/blog/going-fast-with-sqlite-and-python/
    cursor.execute('PRAGMA journal_mode=wal;')
    # 256 MB
    #con.execute('PRAGMA mmap_size=268435456;')
    cursor.execute('PRAGMA mmap_size=16777216;')
    #print(cursor.execute('PRAGMA compile_options').fetchall())
    print(cursor.execute('SELECT sqlite_version()').fetchone())


    cursor.execute(f"CREATE TABLE IF NOT EXISTS {db.DOCUMENT}(\
                {db.ID} INTEGER PRIMARY KEY ASC\
                ,{db.FILE_NAME} TEXT\
                );")
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {db.PASSAGE_GROUP}(\
                {db.ID} INTEGER PRIMARY KEY ASC\
                ,{db.NAME} TEXT\
                ,{db.DOC_ID} INTEGER REFERENCES {db.DOCUMENT}({db.ID})\
                );")
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {db.PASSAGE}(\
                    {db.ID} INTEGER PRIMARY KEY ASC\
                    ,{db.CONTENT} TEXT\
                    ,{db.GROUP_ID} INTEGER REFERENCES {db.PASSAGE_GROUP}({db.ID})\
                    ,{db.PREV_ID} INTEGER REFERENCES {db.PASSAGE}({db.ID}) ON DELETE SET NULL ON UPDATE CASCADE\
                    ,{db.NEXT_ID} INTEGER REFERENCES {db.PASSAGE}({db.ID}) ON DELETE SET NULL ON UPDATE CASCADE\
                );")

    cursor.close()
    con.commit()
    return con



con = open_sqlite();
cursor: sqlite3.Cursor = con.cursor()
has_passage_ids = cursor.execute(f'SELECT rowid,id FROM {db.PASSAGE} ORDER BY rowid DESC LIMIT 1').fetchone() is not None


#embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-small")
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# semantic_splitter = SemanticSplitterNodeParser(
#     buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
# )

# make it small enough that the underlying embedding model can encode it. 
# Note: these are chracters, not tokens. So technically we can make this number bigger
chunk_size=256+64
# chunk_overlap = min(chunk_size / 4, min(chunk_size / 2, 64))
chunk_overlap = 0
paragrgraph_sep = "\n\n\n"
sentence_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, paragraph_separator=paragrgraph_sep)

# reader_web = SimpleWebPageReader(html_to_text=True)
# docs_web = reader_web.load_data(urls = [
#     "file:///Users/jens/Develop/totemic/rag/content/\[Formatted\ for\ LUNA\]\ Heart\ Failure_\ Patient\ Education\ Content_manual.html"
#     ]
# )

reader = SimpleDirectoryReader(input_files=[
    #"content/test.md",
    ],
    filename_as_id=True
)

# markdown_node_parser = MarkdownNodeParser()

cur_docid = 0
group_id = 0
collection:list[str]=[]
document_ids:list[str]=[]
document_metadatas:list[dict[str,Any]]=[]

filename_doc_id_map: dict[str, int] = {}
group_uuid_group_id_map: dict[str, int] = {}
passage_uuid_passage_id_map: dict[str, int] = {}
passage_ids: list[int] = []

for docs in reader.iter_data():
    #semantic_nodes:list[BaseNode] = semantic_splitter.get_nodes_from_documents(docs)
    #markdown_nodes = markdown_node_parser.get_nodes_from_documents(docs)
    base_nodes:list[BaseNode] = sentence_splitter.get_nodes_from_documents(docs)
    for node in base_nodes:
        filename = node.metadata["file_name"]
        if filename in filename_doc_id_map:
            doc_id = filename_doc_id_map[filename]
        else:
            cursor.execute(f'INSERT INTO {db.DOCUMENT} ({db.FILE_NAME}) VALUES (?)'
                           f' RETURNING {db.ID}',
                           (filename, ))
            row = cursor.fetchone()
            (doc_id, ) = row if row else (None, )
            if doc_id is not None:
                filename_doc_id_map[filename] = doc_id

        if node.ref_doc_id is not None:
            if node.ref_doc_id in group_uuid_group_id_map:
                group_id = group_uuid_group_id_map[node.ref_doc_id]
            else:
                cursor.execute(f'INSERT INTO {db.PASSAGE_GROUP} ({db.NAME}, {db.DOC_ID}) VALUES (?, ?)'
                               f' RETURNING {db.ID}',
                               (node.ref_doc_id, doc_id))
                row = cursor.fetchone()
                (group_id, ) = row if row else (None, )
                if group_id:
                    group_uuid_group_id_map[node.ref_doc_id] = group_id

        content: str = node.get_content()
    
        prev_passage_id = passage_uuid_passage_id_map[node.prev_node.node_id] if node.prev_node else None
        cursor.execute(f'INSERT INTO {db.PASSAGE} ({db.ID}, {db.CONTENT}, {db.GROUP_ID}, {db.PREV_ID}) VALUES (?,?,?,?)'
                       f' RETURNING {db.ID}',
                       # Make sure we start passage index at 0, sqlite normaly starts at 1
                       (None if has_passage_ids else 0, content, group_id, prev_passage_id))
        row = cursor.fetchone()
        (passage_id, ) = row if row else (None, )
        if passage_id is not None:
            has_passage_ids = True
            passage_uuid_passage_id_map[node.node_id] = passage_id

            document_ids.append(node.node_id)
            document_metadatas.append({"filename": node.metadata["file_name"],
                                    "docid": doc_id,
                                    "groupid": node.ref_doc_id,
                                    "next": node.next_node.node_id if node.next_node else None})
            collection.append(content)
            passage_ids.append(passage_id)

# add the {db.NEXT_ID}s
cursor.execute(f'UPDATE {db.PASSAGE} AS p'
               f' SET {db.NEXT_ID} = p2.{db.ID}'
               f' FROM {db.PASSAGE} AS p2'
               f' WHERE p2.{db.PREV_ID} = p.{db.ID}'
               f' AND p.{db.NEXT_ID} IS NULL;')


con.commit()

db_collection = DbCollection(db_path=DB_FILE_PATH, cursor=cursor)


# https://jina.ai/news/what-is-colbert-and-late-interaction-and-why-they-matter-in-search/
#RAG = RAGPretrainedModel.from_pretrained("jinaai/jina-colbert-v1-en")
# RAG.index(collection=my_documents, index_name="jina_test_index", max_document_length=8190,)

# full_document = get_wikipedia_page("Hayao_Miyazaki")
# new_documents = get_wikipedia_page("Studio_Ghibli")
# with open("content/[Final Draft for LUNA] Heart Failure_ Patient Education Content.txt") as f:
#     new_documents = f.read()

start_index: float = time.time()

colbert = ColBERT("colbert-ir/colbertv2.0")
path_to_index2: str = colbert.index(
    db_collection=DbCollection(db_path=DB_FILE_PATH, cursor=cursor),
    index_name=INDEX_NAME, 
    max_document_length=chunk_size,
    overwrite=True
)

# RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
# path_to_index2: str = RAG.index(
#     collection=collection, 
#     document_ids=document_ids,
#     document_metadatas=document_metadatas,
#     index_name=INDEX_NAME_RAGA, 
#     max_document_length=chunk_size, 
#     )
# print(path_to_index2)

# RAG: RAGPretrainedModel = RAGPretrainedModel.from_index(INDEX_PATH_RAGA)
# RAG.add_to_index(
#     new_collection=collection, 
#     new_document_ids=document_ids,
#     new_document_metadatas=document_metadatas,
# )


elapsed_index: float = (time.time() - start_index)
print(elapsed_index)

cursor.close()
con.close()






