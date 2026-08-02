# Build PubMed & Wikipedia trên Linux GPU (thay vì Colab)

StatPearls và Textbooks đã build xong trên Colab (nằm trên Google Drive).
PubMed và Wikipedia quá lâu để chạy trên Colab nên build tiếp trên máy Linux
GPU, lưu **local** trên máy này (gộp lại với StatPearls/Textbooks sau, không
cần làm ngay bây giờ). Máy Linux không mount Drive nên không gặp các lỗi FUSE
đã thấy trên Colab — `data/download.py` giờ dùng `huggingface_hub.snapshot_download`
thay cho `git clone`, không phụ thuộc Drive hay git-lfs nữa.

## 0. Yêu cầu

- GPU NVIDIA + driver đã cài (`nvidia-smi` chạy được).
- Python 3.10+, git, `tmux`.
- Dung lượng ổ đĩa trống **≥ 200 GB** cho corpus text + index của PubMed
  (23.9M snippet) + Wikipedia (29.9M snippet). Kiểm tra trước: `df -h .`

## 1. Clone repo & cài môi trường

```bash
cd /workspace
git clone https://github.com/ursuswh-metamorphic/Rag_Router_Reproduce.git fedrag
cd /workspace/fedrag

deactivate 2>/dev/null || true
deactivate 2>/dev/null || true

unset PYTHONHOME
unset PYTHONPATH

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

python -m pip uninstall -y torch torchvision torchaudio
python -m pip cache purge
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Kiểm tra  GPU:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

Nếu `False`, cài lại torch đúng bản CUDA của máy theo hướng dẫn tại
pytorch.org (ví dụ CUDA 12.1: `pip install torch --index-url https://download.pytorch.org/whl/cu121`).

## 2. Chạy build trong `tmux` — bắt buộc để sống sót khi tắt console

`nohup ...&` có thể vẫn bị kill trong một số trường hợp (terminal của IDE,
SSH client cấu hình khác thường...). Cách chắc chắn nhất là chạy trong một
session `tmux` độc lập với phiên đăng nhập: đóng hẳn terminal/tắt SSH, tiến
trình vẫn chạy tiếp trên máy, chỉ cần máy không tắt/reboot.

```bash
# Cài tmux nếu chưa có
sudo apt-get update && sudo apt-get install -y tmux

# Tạo session mới, đặt tên để dễ tìm lại
tmux new -s fedrag_build
```

**Bên trong session tmux** (sau lệnh trên, bạn đang ở trong session mới):

```bash
cd fedrag
source .venv/bin/activate

export FEDRAG_CORPUS_DIR=/data/fedrag_corpus
export FEDRAG_BQ_OVERSAMPLE_FACTOR=3     # rescore BQ lấy knn*3 ứng viên
export FEDRAG_ENABLE_AMP=1               # fp16 khi encode MedCPT
export HF_TOKEN=<token của bạn>          # optional, tránh rate-limit HF Hub
# FEDRAG_FAISS_SHARD_FILES: để mặc định (25) nếu máy có nhiều RAM hơn Colab (>16GB);
# hạ xuống (vd 10) nếu RAM hạn chế.

python -m data.prepare \
    --datasets pubmed wikipedia \
    --index_num_chunks 0 \
    --storage_dir "$FEDRAG_CORPUS_DIR" \
    --download_workers 2 \
    --batch_size 128 \
    2>&1 | tee build.log
```

`batch_size 128` là điểm khởi đầu — tăng dần (256, 512...) nếu `nvidia-smi`
cho thấy còn nhiều VRAM trống, để encode MedCPT nhanh hơn.

**Detach khỏi session** (tiến trình vẫn chạy nền): nhấn `Ctrl+b` rồi `d`.
Giờ có thể đóng terminal/tắt SSH thoải mái.

## 3. Kiểm tra lại / theo dõi bất cứ lúc nào

```bash
tmux ls                    # xem session còn sống không (fedrag_build: ...)
tmux attach -t fedrag_build  # attach lại để xem trực tiếp
# (attach xong, muốn detach lại thì lại Ctrl+b rồi d — đừng gõ Ctrl+c/exit)

tail -f fedrag/build.log   # xem log mà không cần attach
nvidia-smi
free -h
```

## 4. Kiểm tra sau khi build xong

```bash
cd fedrag
python -c "
from fedrag.retriever import Retriever
r = Retriever(corpus_dir='/data/fedrag_corpus')
for name in ('pubmed', 'wikipedia'):
    res = r.query_faiss_index(name, 'What are the complications of a cardiovascular disease?', knn=3)
    print(name, '->', list(res.keys()))
"
```

Khi nào cần gộp chung với StatPearls/Textbooks (đang trên Drive) thì tải 2
thư mục đó (`fedrag_corpus/statpearls`, `fedrag_corpus/textbooks`, ~0.4 GB
tổng) xuống cùng `/data/fedrag_corpus` trên máy này — nhẹ hơn nhiều so với
upload PubMed/Wikipedia (~170 GB) ngược lên Drive.
