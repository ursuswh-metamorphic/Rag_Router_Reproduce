# Build PubMed & Wikipedia trên Linux GPU (thay vì Colab)

StatPearls và Textbooks đã build xong trên Colab, đang nằm trên Google Drive
(`fedrag_corpus/statpearls`, `fedrag_corpus/textbooks`). PubMed và Wikipedia
quá lâu để chạy trên Colab nên build tiếp trên máy Linux GPU, lưu **local**
(không cần mount Drive — máy này không gặp các lỗi FUSE đã thấy trên Colab
vì `data/download.py` giờ dùng `huggingface_hub.snapshot_download` thay cho
`git clone`, không phụ thuộc Drive hay git-lfs nữa).

## 0. Yêu cầu

- GPU NVIDIA + driver đã cài (`nvidia-smi` chạy được).
- Python 3.10+, git.
- Dung lượng ổ đĩa trống **≥ 200 GB** cho corpus text + index của PubMed
  (23.9M snippet) + Wikipedia (29.9M snippet). Kiểm tra trước: `df -h .`

## 1. Clone repo & cài môi trường

```bash
git clone https://github.com/ursuswh-metamorphic/Rag_Router_Reproduce.git fedrag
cd fedrag

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Kiểm tra torch thấy GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Nếu `False`, cài lại torch đúng bản CUDA của máy theo hướng dẫn tại
pytorch.org (ví dụ CUDA 12.1: `pip install torch --index-url https://download.pytorch.org/whl/cu121`).

## 2. Gộp StatPearls & Textbooks đã build sẵn về máy này (không upload PubMed/Wikipedia ngược lên Drive)

`Retriever` cần cả 4 corpus nằm chung 1 thư mục gốc. StatPearls + Textbooks
rất nhỏ (~0.4 GB tổng) nên tải **xuống** máy Linux rẻ hơn nhiều so với việc
upload PubMed/Wikipedia (~170 GB) ngược lên Drive.

```bash
mkdir -p /data/fedrag_corpus
```

Dùng `rclone` (khuyên dùng, xử lý tốt Drive riêng tư qua OAuth):

```bash
# Cài rclone: curl https://rclone.org/install.sh | sudo bash
rclone config                     # chọn "Google Drive", làm theo hướng dẫn OAuth (dán link vào trình duyệt bất kỳ nếu máy không có GUI)
rclone copy gdrive:fedrag_corpus/statpearls /data/fedrag_corpus/statpearls -P
rclone copy gdrive:fedrag_corpus/textbooks  /data/fedrag_corpus/textbooks  -P
```

(Không có rclone thì tải 2 thư mục này qua giao diện web Drive rồi `scp`/`rsync`
lên máy — chỉ ~0.4 GB nên cách nào cũng nhanh.)

## 3. Cấu hình & build PubMed + Wikipedia

```bash
export FEDRAG_CORPUS_DIR=/data/fedrag_corpus
export FEDRAG_BQ_OVERSAMPLE_FACTOR=3     # rescore BQ lấy knn*3 ứng viên
export FEDRAG_ENABLE_AMP=1               # fp16 khi encode MedCPT
export HF_TOKEN=<token của bạn>          # optional, tránh rate-limit HF Hub
# FEDRAG_FAISS_SHARD_FILES: để mặc định (25) nếu máy có nhiều RAM hơn Colab (>16GB);
# hạ xuống (vd 10) nếu RAM hạn chế.

nohup python -m data.prepare \
    --datasets pubmed wikipedia \
    --index_num_chunks 0 \
    --storage_dir "$FEDRAG_CORPUS_DIR" \
    --download_workers 2 \
    --batch_size 128 \
    > build.log 2>&1 &

echo "PID: $!"
```

`batch_size 128` là điểm khởi đầu — tăng dần (256, 512...) nếu `nvidia-smi`
cho thấy còn nhiều VRAM trống, để encode MedCPT nhanh hơn.

## 4. Theo dõi

```bash
tail -f build.log
watch -n 5 nvidia-smi
free -h
```

`nohup` sống sót qua việc mất kết nối SSH (đóng terminal) — muốn chắc chắn
hơn nữa thì chạy cả lệnh trong `tmux`/`screen` để có thể attach lại bất cứ lúc nào.

## 5. Kiểm tra sau khi xong

```bash
python -c "
from fedrag.retriever import Retriever
r = Retriever(corpus_dir='/data/fedrag_corpus')
for name in ('statpearls', 'textbooks', 'pubmed', 'wikipedia'):
    res = r.query_faiss_index(name, 'What are the complications of a cardiovascular disease?', knn=3)
    print(name, '->', list(res.keys()))
"
```

Cả 4 corpus giờ nằm chung `/data/fedrag_corpus` trên máy Linux này — dùng
thẳng thư mục này làm `--storage_dir`/`corpus_dir` cho các bước tiếp theo của
pipeline (FL simulation, router training...) mà không cần đụng tới Drive nữa.
