"""fedrag: A Flower Federated RAG app."""

import warnings
import shutil
import time

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import os
from collections import OrderedDict

import faiss
import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
FAISS_DEFAULT_CONFIG = os.path.join(DIR_PATH, "retriever.yaml")
DEFAULT_CORPUS_DIR = os.path.abspath(
    os.environ.get("FEDRAG_CORPUS_DIR", os.path.join(DIR_PATH, "../data/corpus"))
)
FAISS_SHARD_DIRNAME = "faiss_shards"
FAISS_MANIFEST_NAME = "faiss_manifest.json"
FAISS_SHARD_FILES = max(1, int(os.environ.get("FEDRAG_FAISS_SHARD_FILES", "25")))
FAISS_USE_IVF = os.environ.get("FEDRAG_FAISS_USE_IVF", "0") == "1"


class FaissIndexBundle:

    def __init__(self, shards):
        self.shards = shards
        self.ntotal = sum(shard["index"].ntotal for shard in shards)

    @classmethod
    def from_dataset(cls, dataset_name, corpus_dir=None):
        base_corpus_dir = os.path.abspath(corpus_dir) if corpus_dir else DEFAULT_CORPUS_DIR
        dataset_dir = os.path.join(base_corpus_dir, f"{dataset_name}")
        manifest_path = os.path.join(dataset_dir, FAISS_MANIFEST_NAME)
        legacy_index_path = os.path.join(dataset_dir, "faiss.index")
        legacy_doc_ids_path = os.path.join(dataset_dir, "all_doc_ids.npy")

        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as infile:
                manifest = json.load(infile)

            shards = []
            for shard_info in manifest.get("shards", []):
                index_path = os.path.join(dataset_dir, shard_info["index_path"])
                doc_ids_path = os.path.join(dataset_dir, shard_info["doc_ids_path"])
                vectors_path = os.path.join(dataset_dir, shard_info["vectors_path"])
                if not os.path.exists(index_path) or not os.path.exists(doc_ids_path):
                    raise RuntimeError(
                        f"Missing FAISS shard files for {dataset_name}: {index_path} or {doc_ids_path}"
                    )
                shards.append(
                    {
                        "index": faiss.read_index(index_path),
                        "doc_ids": np.load(doc_ids_path, allow_pickle=False),
                        "vectors_path": vectors_path if os.path.exists(vectors_path) else None,
                    }
                )
            return cls(shards)

        if os.path.exists(legacy_index_path) and os.path.exists(legacy_doc_ids_path):
            return cls(
                [
                    {
                        "index": faiss.read_index(legacy_index_path),
                        "doc_ids": np.load(legacy_doc_ids_path, allow_pickle=False),
                        "vectors_path": None,
                    }
                ]
            )

        raise RuntimeError("FAISS index is not built yet.")

    def search_topk(self, query_embedding, knn=8):
        query = np.asarray(query_embedding, dtype="float32")
        if query.ndim == 1:
            query = np.expand_dims(query, axis=0)

        candidates = []
        for shard_index, shard in enumerate(self.shards):
            index = shard["index"]
            shard_k = min(knn, index.ntotal)
            if shard_k <= 0:
                continue

            distances, doc_idx = index.search(query, shard_k)
            shard_doc_ids = shard["doc_ids"]
            for distance, idx in zip(distances[0], doc_idx[0]):
                if idx < 0:
                    continue
                candidates.append(
                    (
                        float(distance),
                        shard_index,
                        int(idx),
                        str(shard_doc_ids[int(idx)]),
                    )
                )

        candidates.sort(key=lambda item: item[0])
        return candidates[:knn]

    def search(self, query_embedding, knn=8):
        top_candidates = self.search_topk(query_embedding, knn=knn)
        distances = np.asarray([[item[0] for item in top_candidates]], dtype="float32")
        indices = np.asarray([[item[2] for item in top_candidates]], dtype="int64")
        return distances, indices

    def sample_embeddings(self, max_vectors=2048):
        samples = []
        remaining = max_vectors

        for shard in self.shards:
            if remaining <= 0:
                break

            vectors_path = shard.get("vectors_path")
            if vectors_path and os.path.exists(vectors_path):
                shard_vectors = np.load(vectors_path, allow_pickle=False)
                take = min(remaining, len(shard_vectors))
                if take > 0:
                    samples.append(np.asarray(shard_vectors[:take], dtype="float32"))
                    remaining -= take
                continue

            index = shard["index"]
            take = min(remaining, index.ntotal)
            if take <= 0:
                continue

            if hasattr(index, "reconstruct_n"):
                try:
                    samples.append(np.asarray(index.reconstruct_n(0, take), dtype="float32"))
                    remaining -= take
                    continue
                except RuntimeError:
                    pass

            shard_vectors = [index.reconstruct(idx) for idx in range(take)]
            samples.append(np.asarray(shard_vectors, dtype="float32"))
            remaining -= take

        if not samples:
            return np.empty((0, 0), dtype="float32")

        return np.concatenate(samples, axis=0)


class Retriever:

    @staticmethod
    def _dataset_dir(corpus_dir, dataset_name):
        base_corpus_dir = os.path.abspath(corpus_dir) if corpus_dir else DEFAULT_CORPUS_DIR
        return os.path.join(base_corpus_dir, f"{dataset_name}")

    @classmethod
    def _manifest_path(cls, dataset_dir):
        return os.path.join(dataset_dir, FAISS_MANIFEST_NAME)

    @classmethod
    def _shard_dir(cls, dataset_dir):
        return os.path.join(dataset_dir, FAISS_SHARD_DIRNAME)

    @staticmethod
    def _build_index_from_embeddings(embeddings):
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2 or len(embeddings) == 0:
            raise RuntimeError("Cannot build a FAISS index from empty embeddings.")

        d = embeddings.shape[1]
        if not FAISS_USE_IVF or len(embeddings) < 64:
            index = faiss.IndexFlatL2(d)
            index.add(embeddings)
            return index

        nlist = min(max(1, int(np.sqrt(len(embeddings)))), len(embeddings) - 1)
        if nlist < 2:
            index = faiss.IndexFlatL2(d)
            index.add(embeddings)
            return index

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.train(embeddings)
        index.add(embeddings)
        return index

    @classmethod
    def load_index_bundle(cls, dataset_name, corpus_dir=None):
        return FaissIndexBundle.from_dataset(dataset_name, corpus_dir=corpus_dir)

    @classmethod
    def _index_bundle_exists(cls, dataset_name, corpus_dir=None):
        base_corpus_dir = os.path.abspath(corpus_dir) if corpus_dir else DEFAULT_CORPUS_DIR
        dataset_dir = os.path.join(base_corpus_dir, f"{dataset_name}")
        manifest_path = cls._manifest_path(dataset_dir)
        legacy_index_path = os.path.join(dataset_dir, "faiss.index")
        legacy_doc_ids_path = os.path.join(dataset_dir, "all_doc_ids.npy")

        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as infile:
                    manifest = json.load(infile)
                shards = manifest.get("shards", [])
                if not shards:
                    return False
                for shard_info in shards:
                    index_path = os.path.join(dataset_dir, shard_info["index_path"])
                    doc_ids_path = os.path.join(dataset_dir, shard_info["doc_ids_path"])
                    if not os.path.exists(index_path) or not os.path.exists(doc_ids_path):
                        return False
                return True
            except (OSError, json.JSONDecodeError, KeyError):
                return False

        return os.path.exists(legacy_index_path) and os.path.exists(legacy_doc_ids_path)

    def __init__(self, config_file=None, corpus_dir=None):
        if not config_file:
            self.config = yaml.safe_load(open(FAISS_DEFAULT_CONFIG, "r"))
        else:
            self.config = yaml.safe_load(open(config_file, "r"))
        self.corpus_dir = os.path.abspath(corpus_dir) if corpus_dir else DEFAULT_CORPUS_DIR
        self.query_model_name = self.config["query_embedding_model"]
        self.article_model_name = self.config["article_embedding_model"]
        self.emb_dim = self.config["embedding_dimension"]
        self.max_length = self.config.get("max_length", 512)
        self.normalize_embeddings = self.config.get("normalize_embeddings", True)
        self.pooling = self.config.get("pooling", "cls")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.query_tokenizer = AutoTokenizer.from_pretrained(self.query_model_name)
        self.query_encoder = AutoModel.from_pretrained(self.query_model_name).to(
            self.device
        )
        self.query_encoder.eval()

        self.article_tokenizer = AutoTokenizer.from_pretrained(self.article_model_name)
        self.article_encoder = AutoModel.from_pretrained(self.article_model_name).to(
            self.device
        )
        self.article_encoder.eval()

    def _pool_embeddings(self, model_output, attention_mask):
        if self.pooling == "mean":
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()
            )
            pooled = (token_embeddings * input_mask_expanded).sum(1)
            pooled = pooled / input_mask_expanded.sum(1).clamp(min=1e-9)
            return pooled
        return model_output.last_hidden_state[:, 0]

    def _encode_texts(self, texts, tokenizer, encoder, convert_to_numpy=True):
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        encoded_inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded_inputs = {
            key: value.to(self.device) for key, value in encoded_inputs.items()
        }

        with torch.no_grad():
            model_output = encoder(**encoded_inputs)
            embeddings = self._pool_embeddings(
                model_output, encoded_inputs["attention_mask"]
            )
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        if convert_to_numpy:
            embeddings = embeddings.cpu().numpy().astype("float32")
            if single_input:
                return embeddings[0]
            return embeddings

        if single_input:
            return embeddings[0]
        return embeddings

    def encode_query(self, text, convert_to_numpy=True):
        return self._encode_texts(
            text,
            tokenizer=self.query_tokenizer,
            encoder=self.query_encoder,
            convert_to_numpy=convert_to_numpy,
        )

    def encode_articles(self, texts, convert_to_numpy=True):
        return self._encode_texts(
            texts,
            tokenizer=self.article_tokenizer,
            encoder=self.article_encoder,
            convert_to_numpy=convert_to_numpy,
        )

    def build_faiss_index(self, dataset_name, batch_size=32, num_chunks=None):
        dataset_dir = self._dataset_dir(self.corpus_dir, dataset_name)
        chunk_dir = os.path.join(dataset_dir, "chunk")
        shard_dir = self._shard_dir(dataset_dir)
        manifest_path = self._manifest_path(dataset_dir)
        legacy_index_path = os.path.join(dataset_dir, "faiss.index")
        legacy_doc_ids_path = os.path.join(dataset_dir, "all_doc_ids.npy")

        shutil.rmtree(shard_dir, ignore_errors=True)
        os.makedirs(shard_dir, exist_ok=True)
        for path in [manifest_path, legacy_index_path, legacy_doc_ids_path]:
            try:
                os.remove(path)
            except OSError:
                pass

        all_files = sorted(f.path for f in os.scandir(chunk_dir) if f.is_file())
        if num_chunks:
            all_files = all_files[:num_chunks]

        total_files = len(all_files)
        shard_file_limit = max(1, int(os.environ.get("FEDRAG_FAISS_SHARD_FILES", FAISS_SHARD_FILES)))
        processed_files = 0
        processed_docs = 0
        processed_batches = 0
        shard_entries = []
        shard_items = []
        shard_chunk_files = []
        shard_file_count = 0
        shard_started_at = time.time()

        def flush_shard(shard_id: int):
            if not shard_items:
                return None

            valid_items = [
                (doc_id, embedding)
                for doc_id, embedding in shard_items
                if embedding is not None and embedding.shape == (self.emb_dim,)
            ]
            if not valid_items:
                return None

            shard_embeddings = np.asarray([embedding for _, embedding in valid_items], dtype="float32")
            shard_doc_ids = np.asarray([doc_id for doc_id, _ in valid_items])
            shard_index = self._build_index_from_embeddings(shard_embeddings)

            shard_prefix = f"{dataset_name}_shard_{shard_id:04d}"
            index_path = os.path.join(shard_dir, f"{shard_prefix}.faiss.index")
            doc_ids_path = os.path.join(shard_dir, f"{shard_prefix}.doc_ids.npy")
            vectors_path = os.path.join(shard_dir, f"{shard_prefix}.vectors.npy")

            faiss.write_index(shard_index, index_path)
            np.save(doc_ids_path, shard_doc_ids)
            np.save(vectors_path, shard_embeddings)

            shard_elapsed = max(time.time() - shard_started_at, 1e-6)
            shard_rate = len(shard_doc_ids) / shard_elapsed
            tqdm.write(
                f"{dataset_name}: wrote shard {shard_id:04d} with {len(shard_doc_ids)} docs from {len(shard_chunk_files)} chunk files in {shard_elapsed:.1f}s ({shard_rate:.1f} docs/s)"
            )

            return {
                "index_path": os.path.relpath(index_path, dataset_dir),
                "doc_ids_path": os.path.relpath(doc_ids_path, dataset_dir),
                "vectors_path": os.path.relpath(vectors_path, dataset_dir),
                "num_items": int(len(shard_doc_ids)),
                "chunk_files": list(shard_chunk_files),
            }

        shard_id = 0
        for filename in tqdm(all_files, desc=f"{dataset_name}: chunks"):
            batch_content, batch_ids = [], []
            with open(filename, "r", encoding="utf-8") as infile:
                for line in infile:
                    doc = json.loads(line)
                    doc_id = doc.get("id", "")
                    content = doc.get("content", "")
                    batch_ids.append(doc_id)
                    batch_content.append(content)

                    if len(batch_ids) >= batch_size:
                        batch_embeddings = self.encode_articles(
                            batch_content, convert_to_numpy=True
                        )
                        if batch_embeddings.ndim == 1:
                            batch_embeddings = np.expand_dims(batch_embeddings, axis=0)
                        shard_items.extend(zip(batch_ids, batch_embeddings))
                        processed_docs += len(batch_ids)
                        processed_batches += 1
                        tqdm.write(
                            f"{dataset_name}: processed {processed_docs} docs in {processed_batches} batches"
                        )
                        batch_content, batch_ids = [], []

                if batch_content:
                    batch_embeddings = self.encode_articles(
                        batch_content, convert_to_numpy=True
                    )
                    if batch_embeddings.ndim == 1:
                        batch_embeddings = np.expand_dims(batch_embeddings, axis=0)
                    shard_items.extend(zip(batch_ids, batch_embeddings))
                    processed_docs += len(batch_ids)
                    processed_batches += 1
                    tqdm.write(
                        f"{dataset_name}: processed {processed_docs} docs in {processed_batches} batches"
                    )

            shard_file_count += 1
            shard_chunk_files.append(os.path.relpath(filename, dataset_dir))
            processed_files += 1
            tqdm.write(f"{dataset_name}: finished {processed_files}/{total_files} chunk files")

            if shard_file_count >= shard_file_limit:
                shard_entry = flush_shard(shard_id)
                if shard_entry:
                    shard_entries.append(shard_entry)
                    shard_id += 1
                shard_items = []
                shard_chunk_files = []
                shard_file_count = 0
                shard_started_at = time.time()

        if shard_items:
            shard_entry = flush_shard(shard_id)
            if shard_entry:
                shard_entries.append(shard_entry)

        if not shard_entries:
            raise RuntimeError(f"No embeddings were built for corpus {dataset_name}.")

        with open(manifest_path, "w", encoding="utf-8") as outfile:
            json.dump({"format": 1, "dataset": dataset_name, "shards": shard_entries}, outfile, indent=2)

        return

    def query_faiss_index(self, dataset_name, query, knn=8):
        dataset_dir = self._dataset_dir(self.corpus_dir, dataset_name)
        bundle = self.load_index_bundle(dataset_name, corpus_dir=self.corpus_dir)

        query_embedding = self.encode_query(query, convert_to_numpy=True)
        top_candidates = bundle.search_topk(query_embedding, knn=knn)

        chunk_dir = os.path.join(dataset_dir, "chunk")
        final_res = OrderedDict()
        for rank, (doc_score, _, _, doc_id) in enumerate(top_candidates, start=1):
            doc_pref_suf = doc_id.split("_")
            doc_name, snippet_idx = "_".join(doc_pref_suf[:-1]), int(doc_pref_suf[-1])
            full_file = os.path.join(chunk_dir, doc_name + ".jsonl")
            loaded_snippet = json.loads(
                open(full_file).read().strip().split("\n")[snippet_idx]
            )
            final_res[doc_id] = {
                "rank": int(rank),
                "score": float(doc_score),
                "title": str(loaded_snippet["title"]),
                "content": str(loaded_snippet["content"]),
            }

        return final_res

    @classmethod
    def index_exists(cls, dataset_name, corpus_dir=None):
        return cls._index_bundle_exists(dataset_name, corpus_dir=corpus_dir)
