"""fedrag: A Flower Federated RAG app."""

import contextlib
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
# Backwards-compatible alias: some modules (e.g. fedrag/server_app.py) import
# `CORPUS_DIR` directly from this module.
CORPUS_DIR = DEFAULT_CORPUS_DIR
FAISS_SHARD_DIRNAME = "faiss_shards"
FAISS_MANIFEST_NAME = "faiss_manifest.json"
FAISS_SHARD_FILES = max(1, int(os.environ.get("FEDRAG_FAISS_SHARD_FILES", "25")))
FAISS_IVF_MIN_POINTS_PER_CENTROID = max(
    1, int(os.environ.get("FEDRAG_FAISS_IVF_MIN_POINTS_PER_CENTROID", "39"))
)
FAISS_NUM_THREADS = max(
    1, int(os.environ.get("FEDRAG_FAISS_NUM_THREADS", str(os.cpu_count() or 1)))
)
LOG_EVERY_BATCHES = max(1, int(os.environ.get("FEDRAG_LOG_EVERY_BATCHES", "50")))
LOG_EVERY_FILES = max(1, int(os.environ.get("FEDRAG_LOG_EVERY_FILES", "10")))
SAVE_SHARD_VECTORS = os.environ.get("FEDRAG_SAVE_SHARD_VECTORS", "0") == "1"
ENABLE_AMP = os.environ.get("FEDRAG_ENABLE_AMP", "1") == "1"
AMP_DTYPE = os.environ.get("FEDRAG_AMP_DTYPE", "fp16").lower()
ENABLE_TF32 = os.environ.get("FEDRAG_ENABLE_TF32", "1") == "1"
ENABLE_TORCH_COMPILE = os.environ.get("FEDRAG_TORCH_COMPILE", "0") == "1"

# --- Index-technique dispatch -------------------------------------------------
# Small corpora (StatPearls, Textbooks) use 8-bit Scalar Quantization (SQ8):
# a 4x compression (float32 -> int8 codes) that keeps exact-enough L2 search
# entirely inside FAISS with no extra rescoring step.
SQ8_DATASETS = {"statpearls", "textbooks"}
# Large corpora (PubMed, Wikipedia) use Binary Quantization (BQ) + rescoring:
# a 32x compression (768 float32 dims -> 768 packed bits) that keeps the
# *entire* corpus resident in RAM, with a disk-backed float32 memmap used to
# recover exact L2 distances for the shortlisted candidates only.
BQ_DATASETS = {"pubmed", "wikipedia"}
# How many extra candidates (relative to `knn`) stage 1 (binary/Hamming scan)
# fetches before stage 2 rescoring narrows back down to `knn`. Oversampling
# compensates for the recall loss introduced by binarizing vectors.
BQ_OVERSAMPLE_FACTOR = max(1, int(os.environ.get("FEDRAG_BQ_OVERSAMPLE_FACTOR", "3")))
# Cap on how many vectors are used to *train* an SQ8/IVF codebook. Training on
# the full shard is unnecessary and slow once a shard has more than a few
# hundred thousand vectors; a random subsample gives an equally good codebook.
FAISS_SQ8_MAX_TRAIN_POINTS = max(
    256, int(os.environ.get("FEDRAG_SQ8_MAX_TRAIN_POINTS", "200000"))
)
FAISS_SQ8_NPROBE = max(1, int(os.environ.get("FEDRAG_SQ8_NPROBE", "8")))


class FaissIndexBundle:
    """
    In-memory handle over one or more on-disk FAISS shards for a single corpus.

    Each shard is stored on disk as one of two independent "kinds", chosen at
    build time by :meth:`Retriever._dataset_index_kind` based on the corpus
    name:

    - ``"sq8"`` (small corpora: StatPearls, Textbooks) -- an
      ``IndexIVFScalarQuantizer``/``IndexScalarQuantizer`` using
      ``ScalarQuantizer.QT_8bit`` + ``METRIC_L2``. Vectors are compressed
      4x (float32 -> int8 codes) and FAISS returns already-accurate L2
      distances directly, so no extra rescoring step is required.
    - ``"bq"`` (large corpora: PubMed, Wikipedia) -- an ``IndexBinaryFlat``
      built from 1-bit-per-dimension codes (32x compression), plus a
      memory-mapped float32 sidecar file holding the *original* vectors on
      disk. Because Hamming distance over binary codes is only an
      approximation of true L2 distance, queries against a "bq" shard use a
      two-stage oversample + rescore search (see :meth:`_search_bq_shard`).

    A bundle may freely mix shards of both kinds (e.g. while re-indexing a
    dataset with a different technique); each shard carries its own "kind" so
    the query path dispatches per-shard.
    """

    def __init__(self, shards):
        self.shards = shards
        self.ntotal = sum(shard["index"].ntotal for shard in shards)

    @staticmethod
    def _open_vectors_memmap(vectors_path, meta_path):
        """
        Open the float32 rescore sidecar for a "bq" shard as a `np.memmap`.

        The sidecar is stored as a *raw* binary file (not `.npy`) with shape
        and dtype recorded separately in ``meta_path`` so it can be mapped
        with an explicit `np.memmap(..., mode="r")` call, as requested. Using
        `mode="r"` means pages are only pulled from disk (or from Google
        Drive, if `vectors_path` lives on a mounted Drive folder) lazily, on
        first access to each row -- exactly the rows touched during rescore.
        """
        with open(meta_path, "r", encoding="utf-8") as infile:
            meta = json.load(infile)
        shape = tuple(meta["shape"])
        dtype = np.dtype(meta.get("dtype", "float32"))
        return np.memmap(vectors_path, dtype=dtype, mode="r", shape=shape)

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
                # Manifests written before this dispatch existed have no
                # "kind" key; they were always plain/IVF float indexes, which
                # `faiss.read_index` still loads correctly.
                kind = shard_info.get("kind", "sq8")
                index_path = os.path.join(dataset_dir, shard_info["index_path"])
                doc_ids_path = os.path.join(dataset_dir, shard_info["doc_ids_path"])
                if not os.path.exists(index_path) or not os.path.exists(doc_ids_path):
                    raise RuntimeError(
                        f"Missing FAISS shard files for {dataset_name}: {index_path} or {doc_ids_path}"
                    )

                shard = {
                    "kind": kind,
                    "doc_ids": np.load(doc_ids_path, allow_pickle=False),
                    "vectors_path": None,
                    "vectors_mmap": None,
                }

                if kind == "bq":
                    shard["index"] = faiss.read_index_binary(index_path)
                    vectors_rel_path = shard_info.get("vectors_path")
                    vectors_meta_rel_path = shard_info.get("vectors_meta_path")
                    if not vectors_rel_path or not vectors_meta_rel_path:
                        raise RuntimeError(
                            f"BQ shard for {dataset_name} is missing its float32 "
                            "rescore sidecar (vectors_path/vectors_meta_path)."
                        )
                    vectors_path = os.path.join(dataset_dir, vectors_rel_path)
                    vectors_meta_path = os.path.join(dataset_dir, vectors_meta_rel_path)
                    if not os.path.exists(vectors_path) or not os.path.exists(vectors_meta_path):
                        raise RuntimeError(
                            f"Missing BQ rescore sidecar for {dataset_name}: "
                            f"{vectors_path} or {vectors_meta_path}"
                        )
                    shard["vectors_mmap"] = cls._open_vectors_memmap(
                        vectors_path, vectors_meta_path
                    )
                else:
                    shard["index"] = faiss.read_index(index_path)
                    vectors_rel_path = shard_info.get("vectors_path")
                    if vectors_rel_path:
                        candidate = os.path.join(dataset_dir, vectors_rel_path)
                        if os.path.exists(candidate):
                            shard["vectors_path"] = candidate

                shards.append(shard)
            return cls(shards)

        if os.path.exists(legacy_index_path) and os.path.exists(legacy_doc_ids_path):
            return cls(
                [
                    {
                        "kind": "sq8",
                        "index": faiss.read_index(legacy_index_path),
                        "doc_ids": np.load(legacy_doc_ids_path, allow_pickle=False),
                        "vectors_path": None,
                        "vectors_mmap": None,
                    }
                ]
            )

        raise RuntimeError("FAISS index is not built yet.")

    @staticmethod
    def _search_float_shard(shard, shard_index, query, knn):
        """Single-stage exact/approx L2 search for a plain or SQ8 shard."""
        index = shard["index"]
        shard_k = min(knn, index.ntotal)
        if shard_k <= 0:
            return []

        distances, doc_idx = index.search(query, shard_k)
        shard_doc_ids = shard["doc_ids"]
        results = []
        for distance, idx in zip(distances[0], doc_idx[0]):
            if idx < 0:
                continue
            results.append(
                (float(distance), shard_index, int(idx), str(shard_doc_ids[int(idx)]))
            )
        return results

    @staticmethod
    def _search_bq_shard(shard, shard_index, query_vec, knn, oversample=BQ_OVERSAMPLE_FACTOR):
        """
        Two-stage oversample + rescore search for a "bq" (binary-quantized) shard.

        Stage 1 -- RAM, approximate: binarize the query the same way corpus
        vectors were binarized at build time (``bit = 1 if value > 0 else
        0``, packed with `np.packbits`), then run a cheap Hamming-distance
        scan over the fully in-RAM `IndexBinaryFlat` to fetch
        ``knn * oversample`` candidate row ids. This net is deliberately wide
        because 1-bit binarization is lossy and can misrank near neighbours.

        Stage 2 -- disk, exact: use the candidate ids as *random-access* row
        indices into the memory-mapped float32 sidecar file (`shard["vectors_mmap"]`),
        so only the handful of rows actually shortlisted are paged in from
        disk. Their true (squared) L2 distance to the query is computed and
        used to pick the final top-``knn``, correcting for whatever ranking
        error stage 1 introduced.
        """
        binary_index = shard["index"]
        if binary_index.ntotal <= 0:
            return []

        query_bits = np.packbits(query_vec > 0).reshape(1, -1)
        candidate_k = min(max(knn * oversample, knn), binary_index.ntotal)
        if candidate_k <= 0:
            return []

        _, candidate_ids = binary_index.search(query_bits, candidate_k)
        candidate_ids = candidate_ids[0]
        candidate_ids = candidate_ids[candidate_ids >= 0]
        if candidate_ids.size == 0:
            return []

        vectors_mmap = shard["vectors_mmap"]
        # Fancy-indexing a memmap only reads the touched rows off disk.
        candidate_vectors = np.asarray(vectors_mmap[candidate_ids], dtype="float32")
        diffs = candidate_vectors - query_vec
        squared_l2 = np.einsum("ij,ij->i", diffs, diffs)

        order = np.argsort(squared_l2)[:knn]
        shard_doc_ids = shard["doc_ids"]
        return [
            (
                float(squared_l2[pos]),
                shard_index,
                int(candidate_ids[pos]),
                str(shard_doc_ids[int(candidate_ids[pos])]),
            )
            for pos in order
        ]

    def search_topk(self, query_embedding, knn=8):
        query = np.asarray(query_embedding, dtype="float32")
        if query.ndim == 1:
            query = np.expand_dims(query, axis=0)
        query_vec = query[0]

        candidates = []
        for shard_index, shard in enumerate(self.shards):
            if shard["kind"] == "bq":
                candidates.extend(
                    self._search_bq_shard(shard, shard_index, query_vec, knn)
                )
            else:
                candidates.extend(
                    self._search_float_shard(shard, shard_index, query, knn)
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

            if shard["kind"] == "bq":
                # The original float32 vectors are already sitting in the
                # rescore memmap -- reading a prefix from it is cheaper and
                # more accurate than reconstructing from binary codes.
                vectors_mmap = shard["vectors_mmap"]
                take = min(remaining, len(vectors_mmap))
                if take > 0:
                    samples.append(np.asarray(vectors_mmap[:take], dtype="float32"))
                    remaining -= take
                continue

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

            try:
                shard_vectors = [index.reconstruct(idx) for idx in range(take)]
            except RuntimeError:
                continue
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
    def _dataset_index_kind(dataset_name):
        """
        Decide which memory-saving technique a corpus should be indexed with.

        - ``"sq8"``: StatPearls / Textbooks (small corpora) -> 8-bit Scalar
          Quantization (`IndexIVFScalarQuantizer` / `IndexScalarQuantizer`).
        - ``"bq"``: PubMed / Wikipedia (54M+ vectors) -> Binary Quantization
          (`IndexBinaryFlat`) + float32 memmap rescoring.

        Any dataset name outside those two known sets falls back to "sq8"
        (the safer, smaller-footprint default) with a warning, rather than
        silently reintroducing an uncompressed float32 index.
        """
        name = (dataset_name or "").strip().lower()
        if name in BQ_DATASETS:
            return "bq"
        if name in SQ8_DATASETS:
            return "sq8"
        tqdm.write(
            f"Warning: '{dataset_name}' is not a recognized SQ8/BQ dataset name "
            f"(known: {sorted(SQ8_DATASETS | BQ_DATASETS)}); defaulting to the "
            "SQ8 index technique."
        )
        return "sq8"

    @classmethod
    def _training_sample(cls, embeddings):
        """Randomly subsample up to `FAISS_SQ8_MAX_TRAIN_POINTS` rows for training."""
        n = len(embeddings)
        if n <= FAISS_SQ8_MAX_TRAIN_POINTS:
            return embeddings
        sample_idx = np.random.default_rng().choice(
            n, FAISS_SQ8_MAX_TRAIN_POINTS, replace=False
        )
        return embeddings[sample_idx]

    @classmethod
    def _build_sq8_index(cls, embeddings):
        """
        Build an 8-bit Scalar Quantization shard index (float32 -> int8, 4x smaller).

        1. Pick the number of IVF coarse clusters (`nlist`) from the shard
           size, using the same point-per-centroid heuristic as before so
           cluster quality doesn't regress for small shards.
        2. Train the quantizer + coarse centroids on a random *sample* of the
           shard's own embeddings (capped by `FAISS_SQ8_MAX_TRAIN_POINTS`).
        3. Add the *full* set of embeddings -- `index.add` quantizes each
           vector to its 8-bit code as it is inserted, so only the compressed
           codes are kept in RAM/on disk afterwards.
        4. `make_direct_map()` is enabled so `reconstruct`/`reconstruct_n`
           keep working afterwards (used by `sample_embeddings`).

        Falls back to a flat (non-IVF) `IndexScalarQuantizer` when the shard
        is too small to train a meaningful IVF codebook -- still 8-bit
        quantized, just without coarse clustering.
        """
        d = embeddings.shape[1]
        n = len(embeddings)

        nlist = min(max(1, int(np.sqrt(n))), max(1, n - 1)) if n > 1 else 0
        max_nlist_by_points = n // FAISS_IVF_MIN_POINTS_PER_CENTROID
        if max_nlist_by_points > 0:
            nlist = min(nlist, max_nlist_by_points)

        if n < 64 or nlist < 2:
            index = faiss.IndexScalarQuantizer(
                d, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2
            )
            index.train(cls._training_sample(embeddings))
            index.add(embeddings)
            return index

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFScalarQuantizer(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2
        )
        index.train(cls._training_sample(embeddings))
        index.add(embeddings)
        index.make_direct_map()
        index.nprobe = min(nlist, FAISS_SQ8_NPROBE)
        return index

    @staticmethod
    def _build_bq_index(embeddings):
        """
        Build a Binary Quantization shard index (float32 -> 1 bit/dim, 32x smaller).

        Each dimension is binarized with `value > 0`, matching the sign of a
        (roughly zero-centered, L2-normalized) MedCPT embedding, then packed
        8 bits per byte with `np.packbits`. `IndexBinaryFlat` stores those
        packed codes and searches them with a fast, RAM-resident Hamming scan.
        The original float32 vectors are *not* kept by this index -- they are
        written separately to a memmap sidecar for the rescore stage (see
        `Retriever._write_vectors_memmap`).
        """
        d = embeddings.shape[1]
        codes = np.packbits(embeddings > 0, axis=1)
        index = faiss.IndexBinaryFlat(d)
        index.add(codes)
        return index

    @classmethod
    def _build_shard_index(cls, kind, embeddings):
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2 or len(embeddings) == 0:
            raise RuntimeError("Cannot build a FAISS index from empty embeddings.")

        if kind == "bq":
            return cls._build_bq_index(embeddings)
        return cls._build_sq8_index(embeddings)

    @staticmethod
    def _write_vectors_memmap(vectors_path, meta_path, embeddings):
        """
        Persist the original float32 vectors as a `np.memmap` file for BQ rescoring.

        Written as a raw binary blob (rather than `.npy`) with shape/dtype
        recorded in a small JSON sidecar, so it can be reopened later with an
        explicit `np.memmap(path, dtype=..., mode="r", shape=...)` call. This
        is a plain `open()`/`os.ftruncate`-backed file, so it is fully
        compatible with paths on a mounted Google Drive folder -- no special
        cloud APIs are involved, only the file path.
        """
        n, d = embeddings.shape
        mmap_arr = np.memmap(vectors_path, dtype=np.float32, mode="w+", shape=(n, d))
        mmap_arr[:] = embeddings.astype(np.float32, copy=False)
        mmap_arr.flush()
        del mmap_arr  # release the mapping so the file handle is fully closed

        with open(meta_path, "w", encoding="utf-8") as outfile:
            json.dump({"shape": [n, d], "dtype": "float32"}, outfile)

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
                    if shard_info.get("kind") == "bq":
                        vectors_path = shard_info.get("vectors_path")
                        vectors_meta_path = shard_info.get("vectors_meta_path")
                        if not vectors_path or not vectors_meta_path:
                            return False
                        if not os.path.exists(os.path.join(dataset_dir, vectors_path)):
                            return False
                        if not os.path.exists(os.path.join(dataset_dir, vectors_meta_path)):
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
        self.enable_amp = ENABLE_AMP and self.device.type == "cuda"
        self.enable_torch_compile = (
            ENABLE_TORCH_COMPILE and self.device.type == "cuda" and hasattr(torch, "compile")
        )

        if AMP_DTYPE == "bf16":
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16
                if self.device.type == "cuda":
                    tqdm.write("bf16 requested but not supported on this GPU; falling back to fp16.")
        else:
            self.amp_dtype = torch.float16

        if self.device.type == "cuda":
            if ENABLE_TF32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        faiss.omp_set_num_threads(FAISS_NUM_THREADS)

        self.query_tokenizer = None
        self.query_encoder = None

        self.article_tokenizer, self.article_encoder = self._load_encoder(
            self.article_model_name
        )

    def _load_encoder(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name).to(self.device)
        encoder.eval()

        if self.enable_torch_compile:
            try:
                encoder = torch.compile(encoder, mode="max-autotune")
            except Exception as exc:
                tqdm.write(f"Warning: torch.compile disabled for {model_name}: {exc}")

        return tokenizer, encoder

    def _ensure_query_encoder(self):
        if self.query_tokenizer is None or self.query_encoder is None:
            self.query_tokenizer, self.query_encoder = self._load_encoder(
                self.query_model_name
            )

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

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype)
            if self.enable_amp
            else contextlib.nullcontext()
        )
        with torch.inference_mode():
            with autocast_ctx:
                model_output = encoder(**encoded_inputs)
                embeddings = self._pool_embeddings(
                    model_output, encoded_inputs["attention_mask"]
                )
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        embeddings = embeddings.float()

        if convert_to_numpy:
            embeddings = embeddings.cpu().numpy().astype("float32", copy=False)
            if single_input:
                return embeddings[0]
            return embeddings

        if single_input:
            return embeddings[0]
        return embeddings

    def encode_query(self, text, convert_to_numpy=True):
        self._ensure_query_encoder()
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
        """
        Encode a corpus's chunked documents and build its FAISS index shards.

        The corpus is streamed and encoded in shards (as before) to bound
        peak memory, but which *compression technique* each shard is written
        with is now chosen once, up front, from `dataset_name`:

        - StatPearls / Textbooks -> SQ8 (`IndexIVFScalarQuantizer`, int8 codes).
        - PubMed / Wikipedia -> BQ (`IndexBinaryFlat` + float32 memmap rescore
          sidecar), which is what makes it feasible to hold the full ~54.2M
          x 768-dim MedCPT corpus's index in RAM.

        Input/output signature is unchanged: still takes
        `(dataset_name, batch_size, num_chunks)` and returns `None`, writing
        the manifest + shard files to `<corpus_dir>/<dataset_name>/...` (which
        may itself be a path under a mounted Google Drive folder).
        """
        kind = self._dataset_index_kind(dataset_name)
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
            shard_index = self._build_shard_index(kind, shard_embeddings)

            shard_prefix = f"{dataset_name}_shard_{shard_id:04d}"
            doc_ids_path = os.path.join(shard_dir, f"{shard_prefix}.doc_ids.npy")
            np.save(doc_ids_path, shard_doc_ids, allow_pickle=False)

            vectors_rel_path = None
            vectors_meta_rel_path = None
            if kind == "bq":
                # BQ always needs its float32 rescore sidecar -- this is not
                # optional the way SAVE_SHARD_VECTORS is for SQ8 shards.
                index_path = os.path.join(shard_dir, f"{shard_prefix}.binary.index")
                vectors_path = os.path.join(shard_dir, f"{shard_prefix}.vectors.f32.bin")
                vectors_meta_path = os.path.join(shard_dir, f"{shard_prefix}.vectors.meta.json")

                faiss.write_index_binary(shard_index, index_path)
                self._write_vectors_memmap(vectors_path, vectors_meta_path, shard_embeddings)

                vectors_rel_path = os.path.relpath(vectors_path, dataset_dir)
                vectors_meta_rel_path = os.path.relpath(vectors_meta_path, dataset_dir)
            else:
                index_path = os.path.join(shard_dir, f"{shard_prefix}.faiss.index")
                faiss.write_index(shard_index, index_path)

                if SAVE_SHARD_VECTORS:
                    legacy_vectors_path = os.path.join(shard_dir, f"{shard_prefix}.vectors.npy")
                    np.save(legacy_vectors_path, shard_embeddings, allow_pickle=False)
                    vectors_rel_path = os.path.relpath(legacy_vectors_path, dataset_dir)

            shard_elapsed = max(time.time() - shard_started_at, 1e-6)
            shard_rate = len(shard_doc_ids) / shard_elapsed
            tqdm.write(
                f"{dataset_name}: wrote {kind} shard {shard_id:04d} with {len(shard_doc_ids)} docs "
                f"from {len(shard_chunk_files)} chunk files in {shard_elapsed:.1f}s ({shard_rate:.1f} docs/s)"
            )

            return {
                "kind": kind,
                "index_path": os.path.relpath(index_path, dataset_dir),
                "doc_ids_path": os.path.relpath(doc_ids_path, dataset_dir),
                "vectors_path": vectors_rel_path,
                "vectors_meta_path": vectors_meta_rel_path,
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
                        if processed_batches % LOG_EVERY_BATCHES == 0:
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
                    if processed_batches % LOG_EVERY_BATCHES == 0:
                        tqdm.write(
                            f"{dataset_name}: processed {processed_docs} docs in {processed_batches} batches"
                        )

            shard_file_count += 1
            shard_chunk_files.append(os.path.relpath(filename, dataset_dir))
            processed_files += 1
            if processed_files % LOG_EVERY_FILES == 0 or processed_files == total_files:
                tqdm.write(
                    f"{dataset_name}: finished {processed_files}/{total_files} chunk files"
                )

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
        """
        Embed `query` and return the top-`knn` snippets for `dataset_name`.

        Input/output signature is unchanged: `(dataset_name, query, knn)` in,
        an `OrderedDict[doc_id -> {rank, score, title, content}]` out, ranked
        by ascending (squared) L2 distance.

        The compression technique is transparent to the caller -- it is
        resolved per-shard from the on-disk manifest written by
        `build_faiss_index`:

        - SQ8 shards (StatPearls/Textbooks) return exact FAISS L2 distances
          directly from `IndexIVFScalarQuantizer`.
        - BQ shards (PubMed/Wikipedia) run the two-stage oversample + rescore
          search in `FaissIndexBundle._search_bq_shard`: a fast binary/Hamming
          scan over the in-RAM `IndexBinaryFlat` narrows ~54.2M vectors down
          to `knn * 3` candidates, then their true L2 distance is recomputed
          from the on-disk float32 memmap before the final top-`knn` cut.
        """
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
