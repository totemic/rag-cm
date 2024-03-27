#import asyncio
import aiosqlite
import sqlite3
from fastapi import Depends, FastAPI, HTTPException
import time
from typing import Any, Union

# from ragatouillefork import RAGPretrainedModel
from ColBertManager import ColBertManager
from constants import (
    DB_FILE_PATH,
    INDEX_ROOT_PATH,
    INDEX_NAME,
    # INDEX_PATH_RAGA
)
from dbcollection import (open_sqlite_db, DbCollection)

# TODO: replace with async version
con: sqlite3.Connection = open_sqlite_db(DB_FILE_PATH)
cursor: sqlite3.Cursor = con.cursor()
db_collection = DbCollection(db_path=DB_FILE_PATH, cursor=cursor)
colbert_manager = ColBertManager(db_collection, INDEX_ROOT_PATH, INDEX_NAME)

# RAG: RAGPretrainedModel = RAGPretrainedModel.from_index(INDEX_PATH_RAGA)
# results = RAG.search(query="Test", k=3)

# db: aiosqlite.Connection = aiosqlite.connect(DB_FILE)
app = FastAPI()

# Dependency
# Provide a re-usable way to inject a database connection into the individual
# endpoints.
async def get_db():
    """Return a database connection for use as a dependency.
    This connection has the Row row factory automatically attached."""

    db = await aiosqlite.connect(DB_FILE_PATH)
    # Provide a smarter version of the results. This keeps from having to unpack
    # tuples manually.
    db.row_factory = aiosqlite.Row

    try:
        yield db
    finally:
        await db.close()

@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None) -> dict[str, int | Any]:
    return {"item_id": item_id, "q": q}

async def db_test(conn: aiosqlite.Connection) -> list[Any]:
    res: list[Any] = []
    #await asyncio.sleep(5)
    async with conn.execute("SELECT year, title FROM movie ORDER BY year") as cursor:
        async for row in cursor:
            #row.keys
            res.append({'year': row['year'], 'title': row['title']})
    return res

@app.get("/query")
async def query(q: str, count: int=3) -> Any | list[Any]:
    start_search = time.time()
    results = colbert_manager.search(query=q, k=count)
    # results = RAG.search(query=q, k=count)
    elapsed_search = (time.time() - start_search)
    print(elapsed_search)
    return results

@app.get("/querydb")
async def query_db(q: str, count: int | None = None, db_connection: aiosqlite.Connection = Depends(get_db)) -> list[Any]:
    res = await db_test(db_connection)
    return res

# @app.get("/query")
# async def query(q: str, count: int | None = None) -> list[Any]:
#     db_con = await db
#     db_con.row_factory = aiosqlite.Row
#     res = await db_test(db_con)
#     return res

# Route to add a item
@app.post("/items/{item_name}/{quantity}")
def add_item(item_name: str, quantity: int) -> dict[str, str]:
    if quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be greater than 0.")
    # if item already exists, we'll just add the quantity.
    # get all item names
#     items_ids = {item.item_name: item.item_id if item.item_id is not None else 0 for item in grocery_list.values()}
#     if item_name in items_ids.keys():
#         # get index of item_name in item_ids, which is the item_id
#         item_id = items_ids[item_name]
#         grocery_list[item_id].quantity += quantity
# # otherwise, create a new item
#     else:
#         # generate an ID for the item based on the highest ID in the grocery_list
#         item_id = max(grocery_list.keys()) + 1 if grocery_list else 0
#         grocery_list[item_id] = ItemPayload(
#             item_id=item_id, item_name=item_name, quantity=quantity
#         )

#     return {"item": grocery_list[item_id]}
    return {"message": "OK"}