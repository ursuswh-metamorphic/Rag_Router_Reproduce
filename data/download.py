"""fedrag: A Flower Federated RAG app."""

import os
import subprocess
import tempfile
from pathlib import Path

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
        # download_dir may be a network-backed mount (e.g. Google Drive on Colab)
        # that does not support executing files stored on it. Point git's hook
        # lookup at a local, non-mounted directory so post-checkout/post-merge
        # hooks installed by `git lfs install` never need to exec from the mount.
        hooks_path = os.path.join(tempfile.gettempdir(), "fedrag_git_hooks_disabled")
        git_hooks_override = ["-c", f"core.hooksPath={hooks_path}"]

        # If the corpus already exists, only skip it when the download looks complete.
        if corpus != "statpearls" and os.path.isdir(fullpath):
            if cls._needs_lfs_repair(fullpath):
                os.makedirs(hooks_path, exist_ok=True)
                subprocess.run(
                    ["git", *git_hooks_override, "lfs", "pull"], check=True, cwd=fullpath
                )
            print(f"Downloaded {corpus} corpus at {fullpath}.")
            return fullpath

        # If the path exists for non-LFS corpora but does not need repair,
        # return it as-is.
        if os.path.exists(fullpath):
            return fullpath
        if corpus != "statpearls":
            repo_url = f"https://huggingface.co/datasets/MedRAG/{corpus}"
            clone_env = os.environ.copy()
            clone_env["GIT_LFS_SKIP_SMUDGE"] = "1"
            os.makedirs(hooks_path, exist_ok=True)
            subprocess.run(
                ["git", "clone", *git_hooks_override, repo_url, fullpath],
                check=True,
                env=clone_env,
            )
            # Go to the new directory and pull all large files using the Git LFS extension, and back again
            subprocess.run(
                ["git", *git_hooks_override, "lfs", "pull"], check=True, cwd=fullpath
            )
            if cls._needs_lfs_repair(fullpath):
                raise RuntimeError(
                    f"Corpus download finished but some Git LFS files are still missing or pointer-only at {fullpath}. "
                    "Run git lfs pull again inside the corpus directory or delete the directory and rerun the downloader."
                )
        else:
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
