---
title: Federated RAG (FedRAG)
tags: [fedrag, llm]
dataset: [PubMed, StatPearls, Textbooks, Wikipedia, PubMedQA, BioASQ]
framework: [FAISS, transformers]
---

# Federated Retrieval Augmented Generation (FedRAG)

Large Language Models (LLMs) benefit from Retrieval Augmented Generation (RAG) pipelines, which ground their responses
in external data to improve performance. However, organizations often store data in isolated data silos, constraining
classical RAG approaches that rely on centralized data access. By combining Federated Learning with RAG we can query
data across distributed silos without the need to centrally aggregate data, while respecting data privacy.

> [!NOTE]
> This example uses Flower's Message API which remains a preview feature and subject to change.
> Both `ClientApp` and `ServerApp` operate directly on the [Message](https://flower.ai/docs/framework/ref-api/flwr.common.Message.html)
> and [RecordDict](https://flower.ai/docs/framework/ref-api/flwr.common.RecordDict.html) objects.

## Advanced FedRAG Examples

This example provides the building blocks to develop more advanced Federated RAG pipelines, such as enhancing domain-specific
fine-tuned LLMs [[1]](#ref1), using confidential compute environments for secure document re-ranking and LLM inference
[[2]](#ref2), and applying collaborative ANN searches on encrypted data with homomorphic encryption
and multiplicative caching for improved performance [[3]](#ref3).

## FedRAG Pipeline Overview

The figure below demonstrates an overview of the Federated RAG pipeline.

![image info](_static/FedRAG.png)

Given a user query, the server broadcasts the query to each client. Every client retrieves the relevant (top-k)
documents related to the given query and sends them back to the server. The server merges and ranks the retrieved
documents and passes the re-ranked documents as context to the augmented query prompt submitted to the LLM.

## Setup the Example

### System Prerequisites

Depending on whether you are running on macOS, RHEL, Debian please make sure
that the following packages are already installed in your system `wget`, `git-lfs`.

<details>
<summary> Installation instructions for different OS </summary>

```
# wget is used to download .tar files from the Web
# git-lfs is used to download large files from the Hugging Face respository

# macOS
brew install wget   
brew install git-lfs

# RHEL
yum install wget   
yum install git-lfs

# Ubuntu/Debian
apt install wget
apt install git-lfs
 
# Windows
# If you are on Windows, it is highly recommended to make use of WSL with Ubuntu to run your Flower apps. 
# Then, you can install the packages using the above Ubuntu commands.
# Extra tip: with WSL you can also make use of the NVIDIA GPU in your Windows host.
 
# enable Git LFS in your Git environment
# (holds for all systems)
git lfs install
```

</details>

### Clone the Example

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
        && mv _tmp/examples/fedrag . \
        && rm -rf _tmp \
        && cd fedrag
```

This will create a new directory called `fedrag`.

### Install Dependencies

To install all dependencies required to run the example, from the top-level `fedrag` directory execute the following command:

```bash
pip install -e .
```

### Download & Index Corpus

Before you run the Flower engine, please make sure you have downloaded the corpus we need for document retrieval
and created the respective document indices. To accomplish this, run the following helper bash script:

```bash
./data/prepare.sh
```

By default, the above script will download the `Textbooks` and `StatPearls` corpora and create an index
for each corpus using the first `100` chunks (documents). The processed data will be downloaded under the `data/corpus`
directory. The total required disk space for all the documents of `Textbooks` and `StatPearls` is around `3GBs`.

To download all corpora and create an index for all files, please run the following command:

```bash
./data/prepare.sh --datasets "pubmed" "statpearls" "textbooks" "wikipedia" --index_num_chunks 0
```

The total disk space for the all documents of all four corpora is around `120GBs`.

> [!NOTE]
> Please note that for each corpus, its corresponding index might need exactly the same disk space as the documents being indexed.

For an individualized breakdown of the disk space, number of documents, number of snippets, and the domain of each
corpus, please refer to the [README.md](data/README.md) file under the `data` directory.

For more details regarding how each corpus is downloaded and how the corresponding index is created,
please read the section below as well the previously referenced [README.md](data/README.md).

All corpora used in this work were derived from the MedRAG toolkit [[4]](#ref4).

## Run with Simulation Engine

From the top-level directory for this example, launch the simulation:

```bash
flwr run .
```

## Current Architecture and Runbook (Updated)

The active code path now focuses on **router data generation** (not answer generation by an LLM in the main loop).

### Architecture (Current)

1. `data/prepare.py` downloads corpora and builds FAISS indices using MedCPT article embeddings.
2. `fedrag/client_app.py` receives query messages and retrieves top-k snippets from local FAISS indices.
3. `fedrag/server_app.py` broadcasts each MIRAGE question to all clients, collects per-client retrieval results, and assigns router labels.
4. `RagRoute/RR_metadata.py` derives per-source static features (`centroid`, `num_items`, `density`).
5. `fedrag/server_app.py` writes one JSONL sample per `(question, client)` pair to `RagRoute/router_training_data.jsonl`.
6. `RagRoute/RouterNet.py` contains the downstream model definition and a training helper, but this training step is not auto-invoked by `flwr run .`.

### New End-to-End Commands

Use these commands from the repo root:

1. Install dependencies.

```bash
pip install -e .
```

2. Build corpus + FAISS index (default: `statpearls`, `textbooks`, first 100 chunk files each).

```bash
./data/prepare.sh
```

3. Optional: build full indices for all corpora.

```bash
./data/prepare.sh --datasets pubmed statpearls textbooks wikipedia --index_num_chunks 0
```

4. Optional: enable IVF index construction when building FAISS.

```bash
FEDRAG_FAISS_USE_IVF=1 ./data/prepare.sh --datasets statpearls textbooks --index_num_chunks 100
```

5. Optional: control shard size when building indices.

```bash
FEDRAG_FAISS_SHARD_FILES=10 ./data/prepare.sh --datasets statpearls textbooks --index_num_chunks 100
```

6. Run the router-labeling pipeline.

```bash
flwr run .
```

### Important Runtime Notes

1. `RagRoute/router_training_data.jsonl` is generated by the server loop; this is the main artifact of the current pipeline.
2. Router training is a separate step. The repository currently does not expose a standalone CLI script that automatically loads JSONL and trains `RouterNet` end-to-end.
3. IVF is enabled via `FEDRAG_FAISS_USE_IVF=1`, but very small shards still fall back to flat L2 index creation in `fedrag/retriever.py`.

## Expected Results

The current main flow reports router-labeling statistics (not QA answer accuracy).

At the end of execution you should see:

1. Total processed questions per QA dataset.
2. Mean number of positive router labels per question.
3. Mean retrieval + sample-build time per question.
4. Output JSONL path for RouterNet-ready training samples.

You should also see a final summary line similar to:

```text
Saved <N> router training samples to: <path>/RagRoute/router_training_data.jsonl
```

## FedRAG Pipeline Description

### Corpus, Indices & Benchmark Datasets

**Corpus.** The example supports the following corpora for document retrieval:

1. PubMed
2. Textbooks
3. StatPearls
4. Wikipedia

By default, the example uses the `Textbooks` and `StatPearls` corpora.

> [!NOTE]
> The example uses by default the `Textbooks` and `StatPearls` corpora to demonstrate the FedRAG pipeline,
> because the number of documents for `PubMed` and `Wikipedia` are extremely large and downloading and index creation
> can take a lot of time. Please see the instructions [README.md](data/README.md) file on how to
> download the rest of the corpora.

**Index.** For document indexing and retrieval, the example uses the [FAISS](https://github.com/facebookresearch/faiss)
library.

> [!NOTE]
> The example creates by default an index using the first 100 downloaded chunks (i.e., 100 documents).
> We do so in order to quickly create an index for each corpus and bootstrap the example.
> If you want to create an index for all files, please set the `index_num_chunks` flag to `0`.

**QA Datasets.** For QA benchmarking, the example supports the following benchmark datasets:

1. PubMedQA
2. BioASQ
3. MMLU
4. MedQA
5. MedMCQA

By default, the example will evaluate the first `10` questions of the `PubMedQA` and `BioASQ` QA datasets.
To evaluate all the questions from the benchmark dataset, you can disable or comment out the `server-qa-num`
value in the `pyproject.toml` file.

Please see also the section below on how to enable more QA datasets.
All the curated QA benchmark datasets are downloaded from the [MIRAGE](https://github.com/Teddy-XiongGZ/MIRAGE) benchmark [1].

For more details regarding corpus downloading, pre-processing, and indexing steps,
please read the [README.md](../fedrag/data/README.md) file under the `data` directory.

### Document Retrieval and Merge

**Retrieval.** The clients use their local FAISS index to retrieve documents from their local document store.
The `k-nn` value defined in the `[tool.flwr.app.config]` section of the `pyproject.yaml` file controls how many
documents will be retrieved by each client and sent back to the server. The current implementation of document retrieval
for the FAISS index is built with `IndexIVFFlat` and uses the `faiss.METRIC_L2` metric, which means that the lower
the score of a retrieved document the better, since L2 Distance measures dissimilarity.

**Merge.** Once documents and their associated retrieval scores are received by the server, the server merges the retrieved
documents into a single ranked list, either by sorting the documents based on the retrieval score; the lower the score the
more relevant the document is to the query, since we are using the `L2` Euclidean distance. Alternatively, you can use
the simple yet effective Reciprocal Rank Fusion (RRF) method [[5]](#ref5). To smooth ranking differences during merging, using RRF,
you can change the `k-rrf`value defined in the `[tool.flwr.app.config]` section of the `pyproject.yaml` file. Even though
this is a simple merging technique, you should feel free to extend this and define other merging approaches,
such as using a Re-Ranker model.

> [!NOTE]
> If you set `k-rrf=0` then only the retrieval score is considering when merging the retrieved documents,
> while if you set `k-rrf>0` then the retrieved documents are merged using the RRF method.

### Pipeline Configuration

The current example uses the Message API to carry out the communication between the server and the clients. For every
question in the benchmark QA dataset, the server submits the question (query) once to each client and the clients
retrieve the related documents from their respective local document store. Therefore, the server needs only one round
of communication for each question. The properties that are directly related to the execution of the FedRAG application
can be found under the `[tool.flwr.app.config]` section in the `pyproject.yaml` file. These are:

```yaml
server-qa-datasets = ... # the datasets that the server will use to evaluate the FedRAG pipeline
server-qa-num = ... # how many questions should be evaluated per benchmark dataset 
clients-corpus-names = ... # the corpus held by each client participating in the federation environment
k-rrf = ... # the value of the reciprocal rank fusion used by the server to merge the retrieved documents
k-nn = ... # the value of the k nearest neighbors (top-k) documents retrieved at each client and server after merge
server-llm-hfpath = ... # the Hugging Face name/path of the LLM model used by the server to execute the RAG query
```

By default, the current example uses the following two corpora `Textbooks, StatPearls` distributed
across 2 clients, with each client holding one corpus (out of the two). For QA evaluation, the server submits
questions from the following two benchmark QA datasets: `PubMedQA, BioASQ`. For the values
of `k-rrf` and `k-nn`, we use `60` and `8` respectively and for the LLM hosted at the server we use HF's
SmolLM model (`HuggingFaceTB/SmolLM2-1.7B-Instruct`) because for Llama models, we need first to accept the terms.

Specifically, the default values are set as:

```yaml
server-qa-datasets = "pubmedqa|bioasq"
server-qa-num = 10
clients-corpus-names = "Textbooks|StatPearls"
k-rrf = 60
k-nn = 8
server-llm-hfpath = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
```

> [!NOTE]
> The vertical bar in the value of the `server-qa-datasets` is used to pass the name of multiple benchmark
> datasets. Analogously, the vertical bar in the value of the `clients-corpus-names` is used to assign each corpus
> to each client in a Round-Robin fashion, e.g., `Textbooks -> Client 1, StatPearls -> Client 2,  Textbooks-> Client 3,  StatPearls -> Client 4, Textbooks -> Client 5, etc ...`

Based on the computing resources you will use to run the example, please feel free to modify the Hugging Face model path
`server-llm-hfpath` and use a larger model to execute the RAG query. Moreover, if you like, you can perform or introduce
another merging operation at the server-side over the retrieved documents instead of using the simple RRF approach.

### Current Repository Flow

The repository currently contains two related but different flows:

1. The original FedRAG evaluation flow, where the server collected documents from clients, merged them, and then asked an LLM to answer a benchmark question.
2. The current router-labeling flow, where the server uses benchmark questions to generate training data for a routing model instead of generating final answers.

The code under [fedrag/server_app.py](fedrag/server_app.py) now follows the second flow.

### Legacy FedRAG Evaluation Flow

The original flow of this example was:

1. Load benchmark questions from MIRAGE.
2. Broadcast each question to all connected clients.
3. Let each client retrieve top-k documents from its local FAISS index.
4. Merge all retrieved documents at the server.
5. Prompt an LLM with the merged context and benchmark options.
6. Compare predicted answers with gold answers and report accuracy.

That design is still reflected in parts of the older README text above, especially around `server-llm-hfpath`, `k-rrf`, and answer accuracy. The current code path in [fedrag/server_app.py](fedrag/server_app.py) no longer performs answer generation in the main loop.

### Current Router-Labeling Flow

The current runtime flow is:

1. Prepare corpora and FAISS indices under [data](data) using the helper scripts there.
2. Load benchmark QA records from [data/mirage.json](data/mirage.json) through [fedrag/mirage_qa.py](fedrag/mirage_qa.py).
3. Start the Flower simulation with [pyproject.toml](pyproject.toml) providing runtime configuration.
4. In [fedrag/server_app.py](fedrag/server_app.py), wait for available client nodes and assign each node a corpus name.
5. For each benchmark question, send a `MessageType.QUERY` request to every client with `question`, `question_id`, `knn`, `client_slot`, and `corpus_name`.
6. In [fedrag/client_app.py](fedrag/client_app.py), each client opens its assigned FAISS index through [fedrag/retriever.py](fedrag/retriever.py), retrieves top-k documents, and returns `documents`, `scores`, `client_slot`, and `corpus_name`.
7. Back in [fedrag/server_app.py](fedrag/server_app.py), collect per-client retrieval results without flattening away client identity.
8. Build a binary routing label with `labeling_process`: a client receives label `1` if it contributes at least one document to the global top-k after sorting all returned scores; otherwise it receives label `0`.
9. Build a second reference label using [RagRoute/label_query.py](RagRoute/label_query.py), which searches directly over the loaded FAISS indices for consistency checking.
10. Build static source metadata from each FAISS index using [RagRoute/RR_metadata.py](RagRoute/RR_metadata.py): centroid, number of indexed items, and density.
11. Encode the query with the MedCPT query encoder and encode corpus documents with the MedCPT article encoder in [fedrag/retriever.py](fedrag/retriever.py).
12. Concatenate `query_embedding + centroid + distance_to_centroid + num_items + density` into a feature vector compatible with [RagRoute/RouterNet.py](RagRoute/RouterNet.py).
13. Write one JSONL sample per query-client pair to `RagRoute/router_training_data.jsonl`.
14. Print summary statistics: number of processed questions, mean number of positive labels per query, and mean processing time.

### File Responsibilities

The current flow is split across files as follows:

1. [pyproject.toml](pyproject.toml): declares dependencies and Flower runtime config such as benchmark datasets, corpus names, and `k-nn`.
2. [data/prepare.py](data/prepare.py): downloads corpora, builds FAISS indices, and downloads the MIRAGE benchmark JSON.
3. [fedrag/mirage_qa.py](fedrag/mirage_qa.py): loads benchmark records from `mirage.json` and exposes them as an indexable dataset.
4. [fedrag/retriever.py](fedrag/retriever.py): builds FAISS indices, loads them, embeds queries, and retrieves top-k document snippets.
5. [fedrag/client_app.py](fedrag/client_app.py): handles Flower query messages on each client and returns retrieval results plus routing metadata.
6. [fedrag/server_app.py](fedrag/server_app.py): orchestrates the benchmark loop, gathers client replies, assigns routing labels, builds per-source features, and writes JSONL training data.
7. [RagRoute/label_query.py](RagRoute/label_query.py): computes a reference global-top-k source label directly from FAISS searches.
8. [RagRoute/RR_metadata.py](RagRoute/RR_metadata.py): computes source-level metadata features from document embeddings.
9. [RagRoute/RouterNet.py](RagRoute/RouterNet.py): defines the downstream binary routing model expected to consume the generated feature vectors.

### MedCPT Encoder Setup

The current retrieval stack is configured for the MedCPT dual-encoder pair:

1. `ncbi/MedCPT-Article-Encoder` is used when building FAISS indices and when deriving source metadata such as centroids.
2. `ncbi/MedCPT-Query-Encoder` is used for runtime query embedding and for generating router training features.
3. Both encoders are BERT-base models with 768-dimensional outputs, so the RouterNet feature size remains `768 * 2 + 3 = 1539`.

This split is important for RAGRoute because query vectors and corpus vectors are no longer produced by the same model checkpoint.

### Data Produced By The Current Flow

Each line in `RagRoute/router_training_data.jsonl` corresponds to one `(question, client)` pair and contains:

1. Question metadata: dataset name, question id, and question text.
2. Client metadata: `client_slot`, `node_id`, and `corpus_name`.
3. Labels: `label` from live client replies and `reference_label` from direct FAISS search.
4. Features: `feature_vector`, `query_embedding`, `centroid`, `distance_to_centroid`, `num_items`, and `density`.
5. Retrieval trace: `retrieved_documents` and `retrieval_scores`.

### Current Configuration Caveat

The current values in [pyproject.toml](pyproject.toml) list four corpora in `clients-corpus-names`, but the Flower simulation is still configured with `options.num-supernodes = 2`. With the current implementation in [fedrag/server_app.py](fedrag/server_app.py), only the first two client assignments are used when only two nodes are available. If you want all four corpora to participate simultaneously, increase the number of supernodes to `4` or reduce the corpus list to match the number of clients.

Also note that `server-qa-num` is currently commented out in [pyproject.toml](pyproject.toml). That means the server will iterate over all questions in all enabled MIRAGE benchmark datasets, which can take a long time and generate a large JSONL file.

### Enable GPU

In the current router-labeling flow, GPU is mainly useful for transformer embedding on server/client processes.
The main loop does not currently run LLM answer generation.

To allocate GPU to simulation clients, set:

```
options.backend.client-resources.num-gpus = 0.1
```

The value is fractional and can be tuned based on your hardware.

There is still a config key:

```
server-llm-use-gpu = "true"
```

but that flag is part of the legacy answer-generation flow and is not required for the current router data generation loop.

## References

1. <a id="ref1"></a> Jung, Jincheol, Hongju Jeong, and Eui-Nam Huh. "Federated Learning and RAG Integration: A Scalable Approach for Medical Large Language Models." arXiv preprint arXiv:2412.13720 (2024).

2. <a id="ref2"></a> Addison, Parker, Minh-Tuan H. Nguyen, Tomislav Medan, Jinali Shah, Mohammad T. Manzari, Brendan McElrone, Laksh Lalwani, Aboli More, Smita Sharma, Holger R. Roth, Isaac Yang, Chester Chen, Daguang Xu, Yan Cheng, Andrew Feng, and Ziyue Xu. "C-FedRAG: A Confidential Federated Retrieval-Augmented Generation System." arXiv preprint arXiv:2412.13163 (2024).

3. <a id="ref3"></a> Zhao, Dongfang. "FRAG: Toward Federated Vector Database Management for Collaborative and Secure Retrieval-Augmented Generation." arXiv preprint arXiv:2410.13272 (2024).

4. <a id="ref4"></a> Xiong, Guangzhi, Qiao Jin, Zhiyong Lu, and Aidong Zhang. "Benchmarking retrieval-augmented generation for medicine." In Findings of the Association for Computational Linguistics ACL 2024, pp. 6233-6251. 2024.

5. <a id="ref5"></a> Cormack, Gordon V., Charles LA Clarke, and Stefan Buettcher. "Reciprocal rank fusion outperforms condorcet and individual rank learning methods." In Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval, pp. 758-759. 2009.
