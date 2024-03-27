import math
# import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypeVar, Union

import numpy as np
# import srsly
import torch
from colbert import Indexer, IndexUpdater, Searcher, Trainer
from colbert.data import Collection
from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.infra import ColBERTConfig, Run, RunConfig
from colbert.modeling.checkpoint import Checkpoint


#from colbert.data.collection import Collection
from dbcollection import DbCollection


# TODO: Move all bsize related calcs to `_set_bsize()`


class ColBertManager:
    def __init__(
        self,
        db_collection: DbCollection,
        index_root: str,
        index_name: str,
        pretrained_model_path: str | None = None,
        n_gpu: int = -1,
        verbose: int = 1,
        **kwargs:dict[str, Any],
    ):
        self.verbose = verbose

        #self.in_memory_docs = []
        self.base_model_max_tokens = 512
        if n_gpu == -1:
            n_gpu = 1 if torch.cuda.device_count() == 0 else torch.cuda.device_count()

        # hard-code these here. We supply more specific path values througn index_root and index_name which override these in most uses
        root = ".ragindex/"
        experiment = "colbert"
        if pretrained_model_path is None:
            index_path = str(Path(index_root) / index_name)

            ckpt_config = ColBERTConfig.load_from_index(index_path) # type: ignore
            self.config = ckpt_config
            self.run_config = RunConfig(
                nranks=n_gpu, root=root, experiment=experiment
            )
            self.config.index_root = index_root
            self.config.index_name = index_name
            self.checkpoint_name_or_path = self.config.checkpoint

            # TODO: Modify root assignment when loading from HF

        else:
            checkpoint_name_or_path = str(Path(pretrained_model_path))
            ckpt_config = ColBERTConfig.load_from_checkpoint(checkpoint_name_or_path) # type: ignore

            self.run_config = RunConfig(
                nranks=n_gpu, root=root, experiment=experiment
            )
            local_config = ColBERTConfig(**kwargs) # type: ignore
            self.config = ColBERTConfig.from_existing(ckpt_config, local_config,) # type: ignore
            self.config.triples = "unused"
            self.config.queries = "unused"
            self.config.index_root = index_root
            self.config.index_name = index_name
            self.checkpoint_name_or_path: str = checkpoint_name_or_path

        self.db_collection = db_collection
        self.searcher = self.load_and_configure_searcher()

        self.inference_ckpt = Checkpoint(self.checkpoint_name_or_path, colbert_config=self.config)
        self.base_model_max_tokens: int = (self.inference_ckpt.bert.config.max_position_embeddings)

        self.run_context = Run().context(self.run_config)
        self.run_context.__enter__()  # Manually enter the context

    def index(
        self,
        max_document_length: int = 256,
        overwrite: Union[bool, str] = "reuse",
        bsize: int = 32,
    ) -> str:
        if torch.cuda.is_available():
            import faiss

            if not hasattr(faiss, "StandardGpuResources"):
                print(
                    "________________________________________________________________________________\n"
                    "WARNING! You have a GPU available, but only `faiss-cpu` is currently installed.\n",
                    "This means that indexing will be slow. To make use of your GPU.\n"
                    "Please install `faiss-gpu` by running:\n"
                    "pip uninstall --y faiss-cpu & pip install faiss-gpu\n",
                    "________________________________________________________________________________",
                )
                print("Will continue with CPU indexing in 5 seconds...")
                time.sleep(5)

        self.config.doc_maxlen = max_document_length

        # we have added items to the list, make sure db_collection fetches the correct amount next time it calculates the entry count
        passages_count = self.db_collection.read_len()

        nbits = 2
        if passages_count < 5000:
            nbits = 8
        elif passages_count < 10000:
            nbits = 4
        self.config: ColBERTConfig = ColBERTConfig.from_existing(self.config, ColBERTConfig(nbits=nbits, index_bsize=bsize)) # type: ignore

        if passages_count > 100000:
            self.config.kmeans_niters = 4
        elif passages_count > 50000:
            self.config.kmeans_niters = 10
        else:
            self.config.kmeans_niters = 20

        # Instruct colbert-ai to disable forking if nranks == 1
        self.config.avoid_fork_if_possible = True
        self.indexer = Indexer(
            checkpoint=self.checkpoint_name_or_path,
            config=self.config,
            verbose=self.verbose,
        )
        self.indexer.configure(avoid_fork_if_possible=True) # type: ignore


        res_path = self.indexer.index(name=self.config.index_name, collection=self.db_collection, overwrite=overwrite) # type: ignore

        # load searcher right after we created the index
        if self.searcher is None:
            self.searcher = self.load_and_configure_searcher()

        print("Done indexing!")

        return res_path

    def add_to_index(
        self,
        new_passages: list[str],
        new_passage_ids_for_validation: list[int],
        bsize: int = 32,
        allow_reindex:bool = True
    ) -> None:

        # we have added items to the list, make sure db_collection fetches the correct amount next time it calculates the entry count
        combined_len = self.db_collection.read_len()
        new_doc_len = len(new_passages)
        current_len = combined_len - new_doc_len

        if allow_reindex and (current_len + new_doc_len < 5000 or new_doc_len > current_len * 0.05):
            # just reindex the whole collection
            self.index(
                max_document_length=self.config.doc_maxlen,
                overwrite="force_silent_overwrite",
                bsize=bsize,
            )
        else:
            if self.config.index_bsize != bsize:  # Update bsize if it's different
                self.config.index_bsize = bsize

            if self.searcher is None:
                self.searcher = self.load_and_configure_searcher()

            #searcher: Searcher = self.get_searcher_for_index_update()
            updater = IndexUpdater(config=self.config, searcher=self.searcher, 
                                   # TODO: don't specify the checkpoint here, otherwise the model is loaded again
                                   # instead, we manually add the embedder to IndexUpdater
                                   #checkpoint=self.checkpoint_name_or_path
            
            )
            if True:
                # TODO: see above, this code should really live in IndexUpdater by letting us pass an existing Checkpoint
                updater.has_checkpoint = True
                updater.checkpoint = self.searcher.checkpoint # type: ignore
                updater.encoder = CollectionEncoder(config=self.config, checkpoint = self.searcher.checkpoint) # type: ignore

            passage_ids: List[int] = updater.add(new_passages) # type: ignore

            # TODO: compare returned passage_ids with the one we generated ourselves. 
            # If they don't match for some reason, we might have to recreate the entire index

            updater.persist_to_disk()

        print(
            f"Successfully updated index with {new_doc_len} new documents!\n",
            f"New index size: {combined_len}",
        )
    
    def get_next_passage_id_for_insert(self) -> int:
        # TODO: this was copied from IndexUpdater.add
        # ideally this functionality should be exposed by it instead of replicating it here
        start_pid = len(self.searcher.ranker.doclens) if self.searcher is not None else 0
        return start_pid

    
    def delete_from_index(
        self,
        passage_ids: list[int],
    ):
        # Initialize the searcher and updater
        searcher = Searcher(
            checkpoint=self.checkpoint_name_or_path,
            config=None,
            collection=self.collection,
            index=self.config.index_name,
            verbose=self.verbose,
        )
        updater = IndexUpdater(
            config=self.config, searcher=searcher, 
            # don't specifiy checkpoint, otherwise the model will be loaded again
            #checkpoint=self.checkpoint_name_or_path
        )

        # pids_to_remove = []
        # for pid, docid in self.pid_docid_map.items():
        #     if docid in document_ids:
        #         pids_to_remove.append(pid)

        updater.remove(passage_ids)
        updater.persist_to_disk()

        # self.collection = [
        #     doc for pid, doc in enumerate(self.collection) if pid not in pids_to_remove
        # ]
        # self.pid_docid_map = {
        #     pid: docid
        #     for pid, docid in self.pid_docid_map.items()
        #     if pid not in pids_to_remove
        # }

        # if self.docid_metadata_map is not None:
        #     self.docid_metadata_map = {
        #         docid: metadata
        #         for docid, metadata in self.docid_metadata_map.items()
        #         if docid not in document_ids
        #     }

        print(f"Successfully deleted documents with these IDs: {passage_ids}")


    # def get_searcher_for_index_update(self): 
    #     searcher = Searcher(
    #         index_root=self.config.index_root_,
    #         index=self.config.index_name,
    #         checkpoint=self.checkpoint_name_or_path,
    #         config=None,
    #         collection=self.db_collection,
    #         verbose=self.verbose,
    #     )
    #     return searcher


    def load_and_configure_searcher(self) -> Searcher|None:
        # print(
        #     f"Loading searcher for index {self.config.index_name} for the first time...",
        #     "This may take a few seconds",
        # )

        passages_count = self.db_collection.read_len()
        if passages_count == 0:
            # if the database is empty, that means we have no valid index and hence need to postpone loading the searcher until there are entries
            return None
    
        searcher: Searcher = Searcher(
            checkpoint=self.checkpoint_name_or_path,
            config=None,
            collection=self.db_collection,
            #collection=Collection(data=['dummy']), # see TODO below
            index_root=self.config.index_root_,
            index=self.config.index_name,
        )
        # TODO: we can't directly assign it in the constructor - if the db is empty it will treat it as None which crashes
        #searcher.configure(collection=self.db_collection)

        self.configure_searcher(searcher, passages_count)
        return searcher


    #     print("Searcher loaded!")

    def _upgrade_searcher_maxlen(self, maxlen: int):
        if maxlen < 32:
            # Keep maxlen stable at 32 for short queries for easier visualisation
            maxlen = 32
        maxlen = min(maxlen, self.base_model_max_tokens)
        self.searcher.config.query_maxlen = maxlen
        self.searcher.checkpoint.query_tokenizer.query_maxlen = maxlen


    def configure_searcher(self, searcher: Searcher, passages_count: int, force_fast: bool = False):    
        if force_fast:
            # Use fast settingss
            searcher.configure(ncells=1)
            searcher.configure(centroid_score_threshold=0.5)
            searcher.configure(ndocs=256)
        else:
            searcher.configure(ndocs=1024)
            searcher.configure(ncells=16)
            if passages_count < 10000:
                searcher.configure(ncells=8)
                searcher.configure(centroid_score_threshold=0.4)
            elif passages_count < 100000:
                searcher.configure(ncells=4)
                searcher.configure(centroid_score_threshold=0.45)
            else:
                # Otherwise, use defaults for k
                # TODO: add these values in case the config was changed previously
                pass


    def search(
        self,
        query: Union[str, list[str]],
        k: int = 10,
        zero_index_ranks: bool = False,
        passage_ids_to_search: Optional[list[int]] = None
    ) -> list[str] | Any | list[Any]:

        if self.searcher is None:
            print("WARNING: No searcher initialized")
            res: list[str] = []
            return res
    
        base_ncells = self.searcher.config.ncells
        base_ndocs = self.searcher.config.ndocs

        if k > len(self.db_collection):
            print(
                "WARNING: k value is larger than the number of documents in the index!",
                f"Lowering k to {len(self.db_collection)}...",
            )
            k = len(self.db_collection)

        # For smaller collections, we need a higher ncells value to ensure we return enough results
        if k > (32 * self.searcher.config.ncells):
            self.searcher.configure(ncells=min((k // 32 + 2), base_ncells))

        self.searcher.configure(ndocs=max(k * 4, base_ndocs))

        if isinstance(query, str):
            query_length = int(len(query.split(" ")) * 1.35)
            self._upgrade_searcher_maxlen(query_length)
            results = [self._search(query, k, passage_ids_to_search)]
        else:
            longest_query_length = max([int(len(x.split(" ")) * 1.35) for x in query])
            self._upgrade_searcher_maxlen(longest_query_length)
            results = self._batch_search(query, k)

        to_return = []

        for result in results:
            result_for_query = []
            passage_ids: list[int] = [passage_id for passage_id, rank, score in zip(*result)]
            passages = self.db_collection.get_passages_by_id(passage_ids)
            for passage_id, rank, score, passage_res in zip(*result, passages):
                (passage_id1, passage, ) = passage_res
                result_dict = {
                    "content": passage,
                    "score": score,
                    "rank": rank - 1 if zero_index_ranks else rank,
                    "passage_id": passage_id,
                }

                result_for_query.append(result_dict)

            to_return.append(result_for_query)

        # Restore original ncells&ndocs if it had to be changed for large k values
        self.searcher.configure(ncells=base_ncells)
        self.searcher.configure(ndocs=base_ndocs)

        if len(to_return) == 1:
            return to_return[0]
        return to_return

    def _search(self, query: str, k: int, passage_ids_to_search: Optional[list[int]] = None):
        if self.searcher is None:
            return [], [], []
        return self.searcher.search(query, k=k, pids=passage_ids_to_search)

    def _batch_search(self, query: list[str], k: int):
        queries = {i: x for i, x in enumerate(query)}
        results = self.searcher.search_all(queries, k=k)
        results = [
            [list(zip(*value))[i] for i in range(3)]
            for value in results.todict().values()
        ]
        return results

    def train(self, data_dir, training_config: ColBERTConfig):
        training_config = ColBERTConfig.from_existing(self.config, training_config)
        training_config.nway = 2
        with Run().context(self.run_config):
            trainer = Trainer(
                triples=str(data_dir / "triples.train.colbert.jsonl"),
                queries=str(data_dir / "queries.train.colbert.tsv"),
                collection=str(data_dir / "corpus.train.colbert.tsv"),
                config=training_config,
            )

            trainer.train(checkpoint=self.checkpoint_name_or_path)

    def _colbert_score(self, Q, D_padded, D_mask):
        if ColBERTConfig().total_visible_gpus > 0:
            Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

        assert Q.dim() == 3, Q.size()
        assert D_padded.dim() == 3, D_padded.size()
        assert Q.size(0) in [1, D_padded.size(0)]

        scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)
        scores = scores.max(1).values
        return scores.sum(-1)

    def _index_free_search(
        self,
        embedded_queries,
        documents: list[str],
        embedded_docs,
        doc_mask,
        k: int = 10,
        zero_index: bool = False,
    ):
        results = []

        for query in embedded_queries:
            results_for_query = []
            scores = self._colbert_score(query, embedded_docs, doc_mask)
            sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            high_score_idxes = [index for index, _ in sorted_scores[:k]]
            for rank, doc_idx in enumerate(high_score_idxes):
                result = {
                    "content": documents[doc_idx],
                    "score": float(scores[doc_idx]),
                    "rank": rank - 1 if zero_index else rank,
                    "result_index": doc_idx,
                }
                results_for_query.append(result)
            results.append(results_for_query)

        if len(results) == 1:
            return results[0]

        return results

    def _set_inference_max_tokens(
        self, documents: list[str], max_tokens: Union[Literal["auto"], int] = "auto"
    ):
        if (
            not hasattr(self, "inference_ckpt_len_set")
            or self.inference_ckpt_len_set is False
        ):
            if max_tokens == "auto" or max_tokens > self.base_model_max_tokens:
                max_tokens = self.base_model_max_tokens
                percentile_90 = np.percentile(
                    [len(x.split(" ")) for x in documents], 90
                )
                max_tokens = min(
                    math.floor((math.ceil((percentile_90 * 1.35) / 32) * 32) * 1.1),
                    self.base_model_max_tokens,
                )
                max_tokens = max(256, max_tokens)
                if max_tokens > 300:
                    print(
                        f"Your documents are roughly {percentile_90} tokens long at the 90th percentile!",
                        "This is quite long and might slow down reranking!\n",
                        "Provide fewer documents, build smaller chunks or run on GPU",
                        "if it takes too long for your needs!",
                    )
            self.inference_ckpt.colbert_config.max_doclen = max_tokens
            self.inference_ckpt.doc_tokenizer.doc_maxlen = max_tokens
            self.inference_ckpt_len_set = True

    def _index_free_retrieve(
        self,
        query: Union[str, list[str]],
        documents: list[str],
        k: int,
        max_tokens: Union[Literal["auto"], int] = "auto",
        zero_index: bool = False,
        bsize: Union[Literal["auto"], int] = "auto",
    ):
        self._set_inference_max_tokens(documents=documents, max_tokens=max_tokens)

        if k > len(documents):
            print("k value cannot be larger than the number of documents! aborting...")
            return None
        if len(documents) > 1000:
            print(
                "Please note ranking in-memory is not optimised for large document counts! ",
                "Consider building an index and using search instead!",
            )
        if len(set(documents)) != len(documents):
            print(
                "WARNING! Your documents have duplicate entries! ",
                "This will slow down calculation and may yield subpar results",
            )

        embedded_queries = self._encode_index_free_queries(query, bsize=bsize)
        embedded_docs, doc_mask = self._encode_index_free_documents(
            documents, bsize=bsize
        )

        return self._index_free_search(
            embedded_queries=embedded_queries,
            documents=documents,
            embedded_docs=embedded_docs,
            doc_mask=doc_mask,
            k=k,
            zero_index=zero_index,
        )

    def _encode_index_free_queries(
        self,
        queries: Union[str, list[str]],
        bsize: Union[Literal["auto"], int] = "auto",
    ):
        if bsize == "auto":
            bsize = 32
        if isinstance(queries, str):
            queries = [queries]
        maxlen = max([int(len(x.split(" ")) * 1.35) for x in queries])
        self.inference_ckpt.query_tokenizer.query_maxlen = max(
            min(maxlen, self.base_model_max_tokens), 32
        )
        embedded_queries = [
            x.unsqueeze(0)
            for x in self.inference_ckpt.queryFromText(queries, bsize=bsize)
        ]
        return embedded_queries

    def _encode_index_free_documents(
        self,
        documents: list[str],
        bsize: Union[Literal["auto"], int] = "auto",
        verbose: bool = True,
    ):
        if bsize == "auto":
            bsize = 32
            if self.inference_ckpt.doc_tokenizer.doc_maxlen > 512:
                bsize = max(
                    1,
                    int(
                        32
                        / (
                            2
                            ** round(
                                math.log(
                                    self.inference_ckpt.doc_tokenizer.doc_maxlen, 2
                                )
                            )
                            / 512
                        )
                    ),
                )
                print("BSIZE:")
                print(bsize)
        embedded_docs: torch.Tensor = self.inference_ckpt.docFromText(
            documents, bsize=bsize, showprogress=verbose
        )[0]
        doc_mask: torch.Tensor = torch.full(embedded_docs.shape[:2], -float("inf")).to(
            embedded_docs.device
        )
        return embedded_docs, doc_mask

    def rank(
        self,
        query: str,
        documents: list[str],
        k: int = 10,
        zero_index_ranks: bool = False,
        bsize: int = 32,
    ):
        self._set_inference_max_tokens(documents=documents, max_tokens="auto")
        self.inference_ckpt_len_set = False
        return self._index_free_retrieve(
            query, documents, k, zero_index=zero_index_ranks, bsize=bsize
        )

    def encode(
        self,
        documents: list[str],
        document_metadatas: Optional[list[dict]] = None,
        bsize: int = 32,
        max_tokens: Union[Literal["auto"], int] = "auto",
        verbose: bool = True,
    ):
        self._set_inference_max_tokens(documents=documents, max_tokens=max_tokens)
        encodings: torch.Tensor
        doc_masks: torch.Tensor
        encodings, doc_masks = self._encode_index_free_documents(
            documents, bsize=bsize, verbose=verbose
        )
        encodings = torch.cat(
            [
                encodings,
                torch.zeros(
                    (
                        encodings.shape[0],
                        self.inference_ckpt.doc_tokenizer.doc_maxlen
                        - encodings.shape[1],
                        encodings.shape[2],
                    )
                ).to(device=encodings.device),
            ],
            dim=1,
        )
        doc_masks: torch.Tensor = torch.cat(
            [
                doc_masks,
                torch.full(
                    (
                        doc_masks.shape[0],
                        self.inference_ckpt.colbert_config.max_doclen
                        - doc_masks.shape[1],
                    ),
                    -float("inf"),
                ).to(device=doc_masks.device),
            ],
            dim=1,
        )

        if verbose:
            print("Shapes:")
            print(f"encodings: {encodings.shape}")
            print(f"doc_masks: {doc_masks.shape}")

        if hasattr(self, "in_memory_collection"):
            if self.in_memory_metadata is not None:
                if document_metadatas is None:
                    self.in_memory_metadatas.extend([None] * len(documents))
                else:
                    self.in_memory_metadata.extend(document_metadatas)
            elif document_metadatas is not None:
                self.in_memory_metadata = [None] * len(self.in_memory_collection)
                self.in_memory_metadata.extend(document_metadatas)

            self.in_memory_collection.extend(documents)

            # add 0 padding to encodings so they're self.inference_ckpt.doc_tokenizer.doc_maxlen length

            self.in_memory_embed_docs: torch.Tensor = torch.cat(
                [self.in_memory_embed_docs, encodings], dim=0
            )
            self.doc_masks: torch.Tensor = torch.cat([self.doc_masks, doc_masks], dim=0)

        else:
            self.in_memory_collection = documents
            self.in_memory_metadata = document_metadatas
            self.in_memory_embed_docs = encodings
            self.doc_masks = doc_masks

    def search_encoded_docs(
        self,
        queries: Union[str, list[str]],
        k: int = 10,
        bsize: int = 32,
    ):
        queries = self._encode_index_free_queries(queries, bsize=bsize)
        results = self._index_free_search(
            embedded_queries=queries,
            documents=self.in_memory_collection,
            embedded_docs=self.in_memory_embed_docs,
            doc_mask=self.doc_masks,
            k=k,
        )
        if self.in_memory_metadata is not None:
            for result in results:
                result["document_metadata"] = self.in_memory_metadata[
                    result["result_index"]
                ]
        return results

    def clear_encoded_docs(self, force: bool = False):
        if not force:
            print(
                "All in-memory encodings will be deleted in 10 seconds, interrupt now if you want to keep them!"
            )
            print("...")
            time.sleep(10)
        del self.in_memory_collection
        del self.in_memory_metadata
        del self.in_memory_embed_docs
        del self.doc_masks
        del self.inference_ckpt_len_set

    def __del__(self):
        # Clean up context
        try:
            self.run_context.__exit__(None, None, None)
        except Exception:
            print("INFO: Tried to clean up context but failed!")
