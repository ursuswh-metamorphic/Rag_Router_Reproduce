"""fedrag: A Flower Federated RAG app."""

import json
import os
import time
from collections import defaultdict
from time import sleep

import numpy as np
from flwr.app import ConfigRecord, Context, Message, MessageType, RecordDict
from flwr.serverapp import Grid, ServerApp

from RagRoute.RR_metadata import prepare_datasource_metadata
from RagRoute.label_query import generate_labels_for_query
from fedrag.mirage_qa import MirageQA
from fedrag.retriever import CORPUS_DIR, Retriever
from fedrag.task import index_exists

ROUTER_DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "../RagRoute/router_training_data.jsonl"
)
DEFAULT_MIRAGE_PATH = os.path.join(os.path.dirname(__file__), "../data/mirage.json")
MAX_METADATA_VECTORS = 2048


def resolve_repo_path(path_value: str | None, default_path: str) -> str:
    if path_value is None or str(path_value).strip() == "":
        return os.path.abspath(default_path)

    path_value = str(path_value)
    if os.path.isabs(path_value):
        return os.path.abspath(path_value)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.abspath(os.path.join(repo_root, path_value))


def node_online_loop(grid: Grid) -> list[int]:
    node_ids = []
    while not node_ids:
        # Get IDs of nodes available
        node_ids = grid.get_node_ids()
        # Wait if no node is available
        sleep(1)
    return node_ids


def build_node_assignments(node_ids: list[int], corpus_names: list[str]) -> list[dict]:
    assignments = []
    for client_slot, node_id in enumerate(node_ids):
        assignments.append(
            {
                "client_slot": client_slot,
                "node_id": node_id,
                "corpus_name": corpus_names[client_slot % len(corpus_names)],
            }
        )
    return assignments


def load_faiss_bundle(corpus_name: str):
    return Retriever.load_index_bundle(corpus_name, corpus_dir=CORPUS_DIR)


def reconstruct_index_embeddings(index_bundle, max_vectors=MAX_METADATA_VECTORS) -> np.ndarray:
    num_vectors = min(index_bundle.ntotal, max_vectors)
    if num_vectors == 0:
        return np.empty((0, 0), dtype="float32")

    return index_bundle.sample_embeddings(max_vectors=num_vectors)


def build_metadata_store(
    corpus_names: list[str], embedding_dim: int
) -> tuple[dict[str, object], dict[str, dict]]:
    index_by_corpus = {}
    metadata_by_corpus = {}

    for corpus_name in corpus_names:
        if corpus_name in index_by_corpus:
            continue

        index_bundle = load_faiss_bundle(corpus_name)
        embeddings = reconstruct_index_embeddings(index_bundle)
        if embeddings.size == 0:
            metadata = {
                "centroid": np.zeros(embedding_dim, dtype="float32"),
                "num_items": int(index_bundle.ntotal),
                "density": 0.0,
            }
        else:
            metadata = prepare_datasource_metadata(embeddings)
            metadata["centroid"] = np.asarray(metadata["centroid"], dtype="float32")
            metadata["num_items"] = int(index_bundle.ntotal)
            metadata["density"] = float(metadata["density"])

        index_by_corpus[corpus_name] = index_bundle
        metadata_by_corpus[corpus_name] = metadata

    return index_by_corpus, metadata_by_corpus


def build_feature_vector(query_embedding: np.ndarray, metadata: dict) -> np.ndarray:
    centroid = np.asarray(metadata["centroid"], dtype="float32")
    centroid_distance = np.linalg.norm(query_embedding - centroid).astype("float32")
    extra_features = np.array(
        [centroid_distance, metadata["num_items"], metadata["density"]],
        dtype="float32",
    )
    return np.concatenate((query_embedding, centroid, extra_features), axis=0)


def submit_question(
    grid: Grid,
    question: str,
    question_id: str,
    knn: int,
    node_assignments: list[dict],
):
    messages = []
    # Send the same Message to each connected node (which run `ClientApp` instances)
    for assignment in node_assignments:
        # The payload of a Message is of type RecordDict
        # https://flower.ai/docs/framework/ref-api/flwr.common.RecordDict.html
        # which can carry different types of records. We'll use a ConfigRecord object
        # We need to create a new ConfigRecord() object for every node, otherwise
        # if we just override a single key, e.g., corpus_name, the grid will send
        # the same object to all nodes.
        config_record = ConfigRecord()
        config_record["question"] = question
        config_record["question_id"] = question_id
        config_record["knn"] = knn
        config_record["client_slot"] = assignment["client_slot"]
        config_record["corpus_name"] = assignment["corpus_name"]

        task_record = RecordDict({"config": config_record})
        message = Message(
            content=task_record,
            message_type=MessageType.QUERY,  # target `query` method in ClientApp
            dst_node_id=assignment["node_id"],
            group_id=str(question_id),
        )
        messages.append(message)

    # Send messages and wait for all results
    replies = grid.send_and_receive(messages)
    print("Received {}/{} results".format(len(replies), len(messages)))

    client_results = {
        assignment["client_slot"]: {
            "client_slot": assignment["client_slot"],
            "node_id": assignment["node_id"],
            "corpus_name": assignment["corpus_name"],
            "documents": [],
            "scores": [],
        }
        for assignment in node_assignments
    }

    for reply in replies:
        if reply.has_content():
            docs_n_scores = reply.content["docs_n_scores"]
            client_slot = int(docs_n_scores["client_slot"])
            client_results[client_slot] = {
                "client_slot": client_slot,
                "node_id": client_results[client_slot]["node_id"],
                "corpus_name": str(docs_n_scores["corpus_name"]),
                "documents": list(docs_n_scores["documents"]),
                "scores": [float(score) for score in docs_n_scores["scores"]],
            }

    return [client_results[idx] for idx in sorted(client_results)]


def labeling_process(all_client_docs, all_client_scores, k_global=10):
    global_candidates = []

    # 1. Gom tất cả docs từ các client vào một danh sách chung
    for client_id, (docs, scores) in enumerate(zip(all_client_docs, all_client_scores)):
        for doc, score in zip(docs, scores):
            global_candidates.append(
                {
                    "doc": doc,
                    "score": score,
                    "client_id": client_id,
                }
            )

    # 2. Sắp xếp dựa trên score (L2 distance thì sort tăng dần)
    global_candidates.sort(key=lambda x: x["score"])

    # 3. Lấy Top 10 thực tế trên toàn hệ thống
    top_k_global = global_candidates[:k_global]

    # 4. Xác định nhãn cho từng client
    # Nguồn nào đóng góp ít nhất 1 doc vào Top 10 này thì nhãn là 1
    relevant_clients = {item["client_id"] for item in top_k_global}

    num_clients = len(all_client_docs)
    labels = [1 if i in relevant_clients else 0 for i in range(num_clients)]

    return labels


def write_router_samples(output_path: str, samples: list[dict]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as outfile:
        for sample in samples:
            outfile.write(json.dumps(sample) + "\n")


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    node_ids = node_online_loop(grid)

    # k-nearest-neighbors for document retrieval at each client
    knn = int(context.run_config["k-nn"])
    corpus_names = context.run_config["clients-corpus-names"].split("|")
    corpus_names = [c.lower() for c in corpus_names]  # make them lower case
    # Before we start the execution of the FedRAG pipeline,
    # we need to make sure we have downloaded the corpus and
    # created the respective indices
    index_exists(corpus_names)
    node_assignments = build_node_assignments(node_ids, corpus_names)
    qa_datasets = context.run_config["server-qa-datasets"].split("|")
    qa_datasets = [qa_d.lower() for qa_d in qa_datasets]  # make them lower case
    qa_num = context.run_config.get("server-qa-num", None)

    mirage_file_cfg = context.run_config.get(
        "mirage-file-path", os.environ.get("FEDRAG_MIRAGE_FILE")
    )
    mirage_file = resolve_repo_path(mirage_file_cfg, DEFAULT_MIRAGE_PATH)
    if not os.path.exists(mirage_file):
        raise FileNotFoundError(
            f"MIRAGE file was not found: {mirage_file}. "
            "Set run-config 'mirage-file-path' or env 'FEDRAG_MIRAGE_FILE'."
        )
    datasets = {key: MirageQA(key, mirage_file) for key in qa_datasets}

    retriever = Retriever()
    index_by_corpus, metadata_by_corpus = build_metadata_store(
        corpus_names, retriever.emb_dim
    )
    output_path_cfg = context.run_config.get(
        "router-output-path", os.environ.get("FEDRAG_ROUTER_OUTPUT_PATH")
    )
    output_path = resolve_repo_path(output_path_cfg, ROUTER_DATASET_PATH)
    open(output_path, "w", encoding="utf-8").close()
    print(f"Using MIRAGE benchmark file: {mirage_file}")
    print(f"Writing router dataset to: {output_path}")

    question_times = defaultdict(list)
    label_distribution = defaultdict(list)
    sample_count = 0

    for dataset_name in qa_datasets:
        q_idx = 0
        print("Generating router labels for dataset: [{:s}]".format(dataset_name))
        for q in datasets[dataset_name]:
            q_idx += 1
            q_id = f"{dataset_name}_{q_idx}"
            # exit question loop if number of questions has been exceeded
            if qa_num and q_idx > qa_num:
                break

            question = q["question"]
            q_st = time.time()
            client_results = submit_question(
                grid, question, q_id, knn, node_assignments
            )

            all_client_docs = [result["documents"] for result in client_results]
            all_client_scores = [result["scores"] for result in client_results]
            query_embedding = retriever.encode_query(
                question, convert_to_numpy=True
            ).astype("float32")
            labels = labeling_process(all_client_docs, all_client_scores, k_global=knn)
            reference_labels = generate_labels_for_query(
                np.array([query_embedding], dtype="float32"),
                [index_by_corpus[result["corpus_name"]] for result in client_results],
                k=knn,
            )

            samples = []
            for result, label, reference_label in zip(
                client_results, labels, reference_labels
            ):
                metadata = metadata_by_corpus[result["corpus_name"]]
                feature_vector = build_feature_vector(query_embedding, metadata)
                samples.append(
                    {
                        "dataset": dataset_name,
                        "question_id": q_id,
                        "question": question,
                        "client_slot": result["client_slot"],
                        "node_id": result["node_id"],
                        "corpus_name": result["corpus_name"],
                        "label": int(label),
                        "reference_label": int(reference_label),
                        "feature_vector": feature_vector.tolist(),
                        "query_embedding": query_embedding.tolist(),
                        "centroid": metadata["centroid"].tolist(),
                        "distance_to_centroid": float(
                            np.linalg.norm(query_embedding - metadata["centroid"])
                        ),
                        "num_items": int(metadata["num_items"]),
                        "density": float(metadata["density"]),
                        "retrieved_documents": result["documents"],
                        "retrieval_scores": result["scores"],
                    }
                )

            if labels != reference_labels:
                print(
                    f"Warning: reply-based labels {labels} differ from FAISS-only labels {reference_labels} for {q_id}."
                )

            write_router_samples(output_path, samples)
            sample_count += len(samples)

            q_time = time.time() - q_st
            question_times[dataset_name].append(q_time)
            label_distribution[dataset_name].append(sum(labels))

    print(
        "Below, for each benchmark dataset (QA Dataset), we show: \n"
        "(1) the total number of router-labeled queries processed. \n"
        "(2) the mean number of positive client labels per query. \n"
        "(3) the mean wall-clock time to retrieve documents and build router samples. \n"
        f"(4) the JSONL output path containing RouterNet-ready samples: {output_path}.\n"
    )
    for dataset_name in qa_datasets:
        total_questions = len(question_times[dataset_name])
        mean_positive_labels = 0.0
        if label_distribution[dataset_name]:
            mean_positive_labels = float(np.mean(label_distribution[dataset_name]))
        elapsed_time = 0.0
        if question_times[dataset_name]:
            elapsed_time = float(np.mean(question_times[dataset_name]))
        print(
            f"QA Dataset: {dataset_name} \n"
            f"Total Questions: {total_questions} \n"
            f"Mean Positive Labels: {mean_positive_labels} \n"
            f"Mean Querying Time: {elapsed_time} \n"
        )

    print(f"Saved {sample_count} router training samples to: {output_path}")
