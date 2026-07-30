"""fedrag: A Flower Federated RAG app."""

import os
import shutil
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
                if not os.path.isdir(os.path.join(fullpath, ".git")):
                    raise RuntimeError(
                        f"{fullpath} looks incomplete but has no .git directory to repair "
                        "from (git metadata is stripped after a verified download to save "
                        "space). Delete the directory and rerun the downloader so it can "
                        "re-clone from scratch."
                    )
                os.makedirs(hooks_path, exist_ok=True)
                subprocess.run(
                    ["git", *git_hooks_override, "lfs", "pull"],
                    check=True,
                    cwd=fullpath,
                    env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
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
            # Fail fast instead of hanging forever if the LFS remote ever asks
            # for credentials on this (non-interactive, nohup'd) stdin.
            clone_env["GIT_TERMINAL_PROMPT"] = "0"

            # Clone to LOCAL disk first, never straight into download_dir: git and
            # git-lfs writing thousands of small objects directly onto a network
            # mount (e.g. Google Drive on Colab) is unreliable in practice - we've
            # hit both "Permission denied" execing a hook script and "Software
            # caused connection abort" mid-checkout doing this. A local staging
            # clone sidesteps both; only a plain directory move touches the mount.
            staging_root = tempfile.mkdtemp(prefix="fedrag_corpus_staging_")
            staging_path = os.path.join(staging_root, corpus)
            try:
                subprocess.run(
                    ["git", "clone", *git_hooks_override, repo_url, staging_path],
                    check=True,
                    env=clone_env,
                )
                # Go to the new directory and pull all large files using the Git LFS extension, and back again
                subprocess.run(
                    ["git", *git_hooks_override, "lfs", "pull"],
                    check=True,
                    cwd=staging_path,
                    env=clone_env,
                )
                if cls._needs_lfs_repair(staging_path):
                    raise RuntimeError(
                        f"Corpus download finished but some Git LFS files are still missing "
                        f"or pointer-only in the staging clone at {staging_path}."
                    )
                # Drop git metadata before moving onto the (possibly quota-constrained)
                # final destination: .git/lfs/objects duplicates every tracked file
                # already present in the checked-out working tree, roughly doubling
                # on-disk size for no benefit once the download is verified complete.
                shutil.rmtree(os.path.join(staging_path, ".git"), ignore_errors=True)
                if os.path.exists(fullpath):
                    shutil.rmtree(fullpath)
                shutil.move(staging_path, fullpath)
            finally:
                shutil.rmtree(staging_root, ignore_errors=True)
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
