"""fedrag: A Flower Federated RAG app."""

import os
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CORPUS_DIR = os.path.abspath(
    os.environ.get("FEDRAG_CORPUS_DIR", os.path.join(DIR_PATH, "./corpus"))
)


class DownloadCorpora:

    @staticmethod
    def _is_lfs_pointer(file_path: str) -> bool:
        try:
            with open(file_path, "r", encoding="utf-8") as infile:
                return (
                    infile.readline().strip()
                    == "version https://git-lfs.github.com/spec/v1"
                )
        except (OSError, UnicodeDecodeError):
            return False

    @classmethod
    def _needs_lfs_repair(cls, corpus_dir: str) -> bool:
        chunk_dir = os.path.join(corpus_dir, "chunk")
        if not os.path.isdir(chunk_dir):
            return True

        chunk_files = [
            entry.path
            for entry in os.scandir(chunk_dir)
            if entry.is_file() and entry.name.endswith(".jsonl")
        ]
        if not chunk_files:
            return True

        return any(
            os.path.getsize(file_path) == 0 or cls._is_lfs_pointer(file_path)
            for file_path in chunk_files
        )

    @classmethod
    def download(cls, corpus: str, download_dir: str = None) -> str:

        if not download_dir:
            download_dir = CORPUS_DIR
        Path(download_dir).mkdir(parents=True, exist_ok=True)

        fullpath = os.path.join(download_dir, corpus)

        if corpus != "statpearls":
            # If the corpus already exists, only skip it when the download looks complete.
            if os.path.isdir(fullpath) and not cls._needs_lfs_repair(fullpath):
                print(f"Downloaded {corpus} corpus at {fullpath}.")
                return fullpath

            # Fetch each file's real content directly over HTTP via the Hub API
            # instead of `git clone`/`git lfs pull`. git-lfs keeps a full copy of
            # every blob in `.git/lfs/objects` on top of the checked-out working
            # tree (roughly 2x the corpus size on disk while downloading), and we
            # saw it fail outright writing straight onto a Google Drive FUSE mount
            # ("Permission denied" execing a hook, "Software caused connection
            # abort" mid-checkout). snapshot_download writes each file once, with
            # resumable per-file downloads, and works fine against a Drive path.
            snapshot_download(
                repo_id=f"MedRAG/{corpus}",
                repo_type="dataset",
                local_dir=fullpath,
                max_workers=4,
            )
            if cls._needs_lfs_repair(fullpath):
                raise RuntimeError(
                    f"Corpus download finished but some files are still missing or "
                    f"empty at {fullpath}. Delete the directory and rerun the downloader."
                )
        else:
            if os.path.exists(fullpath):
                return fullpath
            # Download directly from the NIH repo
            archive_url = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz"
            subprocess.run(
                ["wget", archive_url, "-P", fullpath],
                check=True,
            )
            archive_path = os.path.join(fullpath, "statpearls_NBK430685.tar.gz")
            subprocess.run(
                ["tar", "-xzvf", archive_path, "-C", fullpath],
                check=True,
            )
            print("Chunking the statpearls corpus...")
            # Use the provided statpearls.py script to split the files into chunks
            chunk_env = os.environ.copy()
            chunk_env["FEDRAG_CORPUS_DIR"] = os.path.abspath(download_dir)
            subprocess.run(
                ["python", os.path.join(DIR_PATH, "statpearls.py")],
                check=True,
                env=chunk_env,
            )
        print(f"Downloaded {corpus} corpus at {fullpath}.")
        return fullpath
