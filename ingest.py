import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# from typing import Any
# import requests
import time
from sqlite3 import (Connection, Cursor)

from constants import (
    DB_FILE_PATH, 
    INDEX_ROOT_PATH,
    INDEX_NAME,
    # INDEX_NAME_RAGA,
    # INDEX_PATH_RAGA
)
import db

# from pathlib import Path
from ColBertManager import ColBertManager
# from ragatouillefork import RAGPretrainedModel


from llama_index.core import SimpleDirectoryReader
# from llama_index.readers.web import SimpleWebPageReader

from llama_index.core.schema import (
    BaseNode,
    # Document,
)
from llama_index.core.node_parser import (
    # MarkdownNodeParser,
    SentenceSplitter,
    # SemanticSplitterNodeParser,
)
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#from colbert.data.collection import Collection
from dbcollection import ( open_sqlite_db, create_tables_if_missing, sql_parameter_marks, DbCollection )

def ingest_documents(con: Connection, input_files: list[str]): 
    cursor: Cursor = con.cursor()
    is_empty_db = db_collection.read_len() == 0
    next_passage_id = 0 if is_empty_db else colbert_manager.get_next_passage_id_for_insert()
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
    paragraph_sep = "\n\n\n"
    sentence_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, paragraph_separator=paragraph_sep)

    reader = SimpleDirectoryReader(input_files=input_files,
        filename_as_id=True
    )

    # markdown_node_parser = MarkdownNodeParser()

    
    # document_ids:list[str]=[]
    # document_metadatas:list[dict[str,Any]]=[]

    collection:list[str]=[]

    filename_doc_id_map: dict[str, int] = {}
    group_uuid_group_id_map: dict[str, int] = {}
    passage_uuid_passage_id_map: dict[str, int] = {}
    passage_ids: list[int] = []

    for docs in reader.iter_data():
        #semantic_nodes:list[BaseNode] = semantic_splitter.get_nodes_from_documents(docs)
        #markdown_nodes = markdown_node_parser.get_nodes_from_documents(docs)
        base_nodes:list[BaseNode] = sentence_splitter.get_nodes_from_documents(docs)
        for node in base_nodes:
            group_id = 0
            filename = node.metadata["file_name"]
            if filename in filename_doc_id_map:
                doc_id = filename_doc_id_map[filename]
            else:
                cursor.execute(f'INSERT INTO {db.DOCUMENT} ({db.FILE_NAME}) VALUES (?)'
                               f' RETURNING {db.ID}',
                               (filename, ))
                row = cursor.fetchone()
                (doc_id, ) = row if row  is not None else (None, )
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
                    (group_id, ) = row if row is not None else (None, )
                    if group_id:
                        group_uuid_group_id_map[node.ref_doc_id] = group_id

            content: str = node.get_content()
        
            prev_passage_id = passage_uuid_passage_id_map[node.prev_node.node_id] if node.prev_node else None
            cursor.execute(f'INSERT INTO {db.PASSAGE} ({db.ID}, {db.CONTENT}, {db.GROUP_ID}, {db.PREV_ID}) VALUES (?,?,?,?)'
                           f' RETURNING {db.ID}',
                           (next_passage_id, content, group_id, prev_passage_id))
            row = cursor.fetchone()
            (passage_id, ) = row if row is not None else (None, )
            if passage_id is not None:
                next_passage_id = next_passage_id + 1
                passage_uuid_passage_id_map[node.node_id] = passage_id

                # document_ids.append(node.node_id)
                # document_metadatas.append({"filename": node.metadata["file_name"],
                #                            "docid": doc_id,
                #                            "groupid": node.ref_doc_id,
                #                            "next": node.next_node.node_id if node.next_node else None})
                collection.append(content)
                passage_ids.append(passage_id)

    # add the {db.NEXT_ID}s
    cursor.execute(f'UPDATE {db.PASSAGE} AS p'
                   f' SET {db.NEXT_ID} = p2.{db.ID}'
                   f' FROM {db.PASSAGE} AS p2'
                   f' WHERE p2.{db.PREV_ID} = p.{db.ID}'
                   f' AND p.{db.NEXT_ID} IS NULL;')

    # https://jina.ai/news/what-is-colbert-and-late-interaction-and-why-they-matter-in-search/
    #RAG = RAGPretrainedModel.from_pretrained("jinaai/jina-colbert-v1-en")
    # RAG.index(collection=my_documents, index_name="jina_test_index", max_document_length=8190,)

    start_index: float = time.time()

    if is_empty_db:
        path_to_index: str = colbert_manager.index(
            max_document_length=chunk_size,
            overwrite=True
        )
        logger.info(f'Created index at {path_to_index}')
    else:
        colbert_manager.add_to_index(
            passages=collection, 
            passage_ids_for_validation=passage_ids
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

    # only commit the results to the db if adding to to ColBert index succeeded
    con.commit()
    cursor.close
    return

def delete_document_passages(con: Connection, passage_ids: list[int]):
    con.execute(f'DELETE FROM {db.PASSAGE} AS p'
                   f' WHERE p.{db.ID} IN ({sql_parameter_marks(passage_ids)})', passage_ids)
    colbert_manager.remove_from_index(passage_ids)
    con.commit()


if __name__ == "__main__":
    con: Connection = open_sqlite_db(DB_FILE_PATH)
    create_tables_if_missing(con)
    cursor: Cursor = con.cursor()

    db_collection = DbCollection(db_path=DB_FILE_PATH, cursor=cursor)
    is_empty_db = db_collection.read_len() == 0
    # is_empty_db = cursor.execute(f'SELECT rowid,id FROM {db.PASSAGE} ORDER BY rowid DESC LIMIT 1').fetchone() is None
    colbert_manager = ColBertManager(db_collection, INDEX_ROOT_PATH, INDEX_NAME, "colbert-ir/colbertv2.0"
                                    ) if is_empty_db else ColBertManager(db_collection, INDEX_ROOT_PATH, INDEX_NAME)
        

    ingest_documents(con, [
            "test/test.md"
            ])

    results = colbert_manager.search(query="what's the best passge for number 18?", k=3)
    print(results)

    ingest_documents(con, [
            "test/test2.md",
            ])

    print(colbert_manager.search(query="what's the best passge for number 18?", k=3))

    delete_document_passages(con, passage_ids=[2, 3])

    print(colbert_manager.search(query="what's the best passge for number 18?", k=3))

    delete_document_passages(con, passage_ids=[18, 16])

    print(colbert_manager.search(query="what's the best passge for number 18?", k=3))

    cursor.close()
    con.close()
