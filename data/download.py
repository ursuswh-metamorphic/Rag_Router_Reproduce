"""fedrag: A Flower Federated RAG app."""

import os
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CORPUS_DIR = os.path.abspath(
    os.environ.get("FEDRAG_CORPUS_DIR", os.path.join(DIR_PATH, "./corpus"))
)

MAX_DOWNLOAD_ATTEMPTS = 5


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

    @staticmethod
    def _expected_chunk_sizes(repo_id: str):
        """Map of {basename: expected byte size} for chunk/*.jsonl on the Hub.

        Returns None (meaning "can't verify, don't block on it") if the Hub
        API call fails, e.g. no network -- a fully on-disk corpus should
        still be usable offline. Batches the paths-info lookup: the Hub's
        endpoint 413s if asked about ~1000+ paths in one request.
        """
        try:
            api = HfApi()
            files = api.list_repo_files(repo_id, repo_type="dataset")
            chunk_paths = [f for f in files if f.startswith("chunk/") and f.endswith(".jsonl")]
            sizes = {}
            batch_size = 200
            for i in range(0, len(chunk_paths), batch_size):
                batch = chunk_paths[i : i + batch_size]
                for info in api.get_paths_info(repo_id, batch, repo_type="dataset"):
                    sizes[os.path.basename(info.path)] = info.size
            return sizes
        except Exception:
            return None

    @classmethod
    def _needs_lfs_repair(cls, corpus_dir: str, repo_id: str = None) -> bool:
        chunk_dir = os.path.join(corpus_dir, "chunk")
        if not os.path.isdir(chunk_dir):
            return True

        chunk_files = {
            entry.name: entry.path
            for entry in os.scandir(chunk_dir)
            if entry.is_file() and entry.name.endswith(".jsonl")
        }
        if not chunk_files:
            return True

        # Compare against the Hub's real per-file sizes when possible. A
        # previous run that got interrupted mid-download (crash, Ctrl-C, a
        # dropped connection) can leave files that are individually
        # nonexistent-or-truncated; comparing sizes catches both a missing
        # file (count mismatch) and a truncated one (size mismatch) in one
        # pass. Crucially this also avoids flagging a file as "broken" just
        # because it's 0 bytes -- some MedRAG chunk files (e.g.
        # pubmed23n0654.jsonl) are genuinely empty on the Hub itself, and no
        # amount of retrying downloads a file that was never there.
        expected_sizes = cls._expected_chunk_sizes(repo_id) if repo_id else None
        if expected_sizes is not None:
            if len(chunk_files) < len(expected_sizes):
                return True
            return any(
                os.path.getsize(path) != expected_sizes[name]
                for name, path in chunk_files.items()
                if name in expected_sizes
            )

        # No repo_id / API unreachable: fall back to the old heuristic. This
        # can't distinguish "empty by design" from "corrupted", so it may
        # false-flag a genuinely-empty chunk file, but a network-down
        # environment shouldn't block on that.
        return any(
            os.path.getsize(path) == 0 or cls._is_lfs_pointer(path)
            for path in chunk_files.values()
        )

    @classmethod
    def download(cls, corpus: str, download_dir: str = None) -> str:

        if not download_dir:
            download_dir = CORPUS_DIR
        Path(download_dir).mkdir(parents=True, exist_ok=True)

        fullpath = os.path.join(download_dir, corpus)
        repo_id = f"MedRAG/{corpus}"

        if corpus != "statpearls":
            # If the corpus already exists, only skip it when the download looks complete.
            if os.path.isdir(fullpath) and not cls._needs_lfs_repair(fullpath, repo_id=repo_id):
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
            #
            # A single pass can still land some files empty/truncated on flaky
            # storage backends. snapshot_download itself re-verifies and
            # re-fetches any file that doesn't match what the repo expects on
            # a subsequent call (confirmed: truncating a file to 0 bytes and
            # calling it again restores the correct content), so just retry a
            # few times in place instead of surfacing a one-shot failure that
            # forces a manual rerun of the whole pipeline.
            for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    local_dir=fullpath,
                    max_workers=4,
                )
                if not cls._needs_lfs_repair(fullpath, repo_id=repo_id):
                    break
                print(
                    f"{corpus}: download attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS} "
                    "still incomplete, retrying fetch..."
                )
            else:
                raise RuntimeError(
                    f"Corpus download still incomplete after {MAX_DOWNLOAD_ATTEMPTS} "
                    f"attempts at {fullpath}. Delete the directory and rerun the downloader."
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
