"""fedrag: A Flower Federated RAG app."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
from pathlib import Path

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.dirname(DIR_PATH)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.download import DownloadCorpora
from fedrag.mirage_qa import MirageQA
from fedrag.retriever import Retriever


VALID_DATASETS = ["pubmed", "statpearls", "textbooks", "wikipedia"]

if __name__ == "__main__":

    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Datasets to download and index for the FedRAG workflow."
    )

    # Add an optional positional argument for number of partitions
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["textbooks"],
        help="List of dataset names as strings. Valid names are: `pubmed`, `textbooks`, `statpearls`, `wikipedia`",
    )

    # Add an optional positional argument for number of partitions
    parser.add_argument(
        "--index_num_chunks",
        nargs="?",
        type=int,
        default=0,
        help="How many chunks to consider when building the index for each corpus.",
    )

    parser.add_argument(
        "--storage_dir",
        nargs="?",
        default=None,
        help="Base directory to store/read corpora and FAISS indexes (e.g., a mounted Google Drive path).",
    )

    parser.add_argument(
        "--download_workers",
        nargs="?",
        type=int,
        default=1,
        help="Number of concurrent workers to use when downloading corpora.",
    )

    parser.add_argument(
        "--batch_size",
        nargs="?",
        type=int,
        default=64,
        help="Number of documents to encode per MedCPT batch when building FAISS indexes.",
    )

    args = parser.parse_args()
    dataset_names = sorted(
        {dataset_name.lower() for dataset_name in args.datasets}
    )  # make sure each dataset appears once in the collection
    num_chunks = (
        None if args.index_num_chunks == 0 else args.index_num_chunks
    )  # set to None if 0 else int value
    storage_dir = os.path.abspath(args.storage_dir) if args.storage_dir else None
    if storage_dir:
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
    download_workers = max(1, args.download_workers)
    batch_size = max(1, args.batch_size)

    # use the default configurations for the FAISS based retrieval system
    retriever = Retriever(corpus_dir=storage_dir)

    sample_query = "What are the complications of a cardiovascular disease?"
    failed_datasets = []

    def download_corpus(dataset_name: str) -> str:
        print(f"Downloading corpus: {dataset_name}")
        DownloadCorpora.download(corpus=dataset_name, download_dir=storage_dir)
        print(f"Downloaded corpus: {dataset_name}")
        return dataset_name

    downloaded_datasets = set()
    valid_dataset_names = [
        dataset_name for dataset_name in dataset_names if dataset_name in VALID_DATASETS
    ]

    if download_workers > 1 and len(valid_dataset_names) > 1:
        with ThreadPoolExecutor(max_workers=download_workers) as executor:
            futures = {
                executor.submit(download_corpus, dataset_name): dataset_name
                for dataset_name in valid_dataset_names
            }
            for future in as_completed(futures):
                dataset_name = futures[future]
                try:
                    future.result()
                    downloaded_datasets.add(dataset_name)
                except Exception as exc:
                    failed_datasets.append((dataset_name, exc))
                    print(f"Failed corpus {dataset_name}: {exc}")
    else:
        for dataset_name in valid_dataset_names:
            try:
                download_corpus(dataset_name)
                downloaded_datasets.add(dataset_name)
            except Exception as exc:
                failed_datasets.append((dataset_name, exc))
                print(f"Failed corpus {dataset_name}: {exc}")

    for dataset_name in dataset_names:
        if dataset_name not in VALID_DATASETS:
            print("Not a valid dataset name: ", dataset_name)
            continue

        if dataset_name not in downloaded_datasets:
            continue

        try:
            print(f"Rebuilding MedCPT FAISS index for corpus: {dataset_name}")
            retriever.build_faiss_index(
                dataset_name=dataset_name,
                batch_size=batch_size,
                num_chunks=num_chunks,
            )
            print(f"Built MedCPT FAISS index for corpus: {dataset_name}")

            print(f"Querying FAISS indexer of corpus: {dataset_name}")
            res = retriever.query_faiss_index(dataset_name, sample_query, knn=2)
            print(
                f"Query: {sample_query} and Corpus: {dataset_name}, returned the following results: {res}"
            )
        except Exception as exc:
            failed_datasets.append((dataset_name, exc))
            print(f"Failed corpus {dataset_name}: {exc}")

    print("Downloading MIRAGE QA benchmark data.")
    mirage_file = os.path.join(DIR_PATH, "mirage.json")
    MirageQA.download(mirage_file)
    print("Downloaded MIRAGE QA benchmark data.")

    if failed_datasets:
        print("Some corpora failed, but the script continued with the remaining datasets:")
        for dataset_name, exc in failed_datasets:
            print(f"- {dataset_name}: {exc}")
