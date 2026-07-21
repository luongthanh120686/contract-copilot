# ============================================================
# CHẶNG 7 — Đóng gói app thành "thùng hàng tiêu chuẩn"
# 📖 Ôn lại: Bài 66 (Docker cơ bản) · Bài 67 (deploy)
#
# Docker giải câu nói kinh điển "trên máy tao chạy được mà":
# đóng cả app + Python + thư viện vào một container, chạy đâu cũng giống nhau.
#
# Build:  docker build -t contract-copilot .
# Chạy:   docker run -p 7860:7860 \
#           -e OLLAMA_HOST=http://host.docker.internal:11434 \
#           contract-copilot
# ============================================================

# Ảnh nền: Python 3.11 bản "slim" (gọn, ~150MB thay vì ~1GB bản đầy đủ)
FROM python:3.11-slim

# Thư mục làm việc bên trong container
WORKDIR /app

# --- BƯỚC QUAN TRỌNG: copy requirements.txt TRƯỚC, copy code SAU ---
# Docker chia image thành từng "lớp" và nhớ lại lớp không đổi.
# Sửa code thì chỉ lớp cuối bị build lại; nếu copy code trước thì
# mỗi lần sửa 1 dòng là phải cài lại toàn bộ thư viện (~10 phút).
COPY requirements.txt .

# BẪY LỚN NHẤT của Dockerfile này: `pip install sentence-transformers` kéo
# theo torch bản đầy đủ, trong đó có driver GPU NVIDIA (nvidia 2,9GB +
# triton 652MB) — VÔ DỤNG vì container này chỉ chạy CPU.
# Cài torch bản CPU-only TRƯỚC để pip không kéo bản CUDA về:
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt
# --no-cache-dir: không giữ file cài tạm → image nhẹ hơn vài trăm MB

# Giờ mới copy code và dữ liệu
COPY src/ ./src/
COPY data/*.txt data/chunks.json data/golden.json data/nhan.json ./data/

# Tải sẵn model embedding vào image (~470MB) để lần chạy đầu không phải
# chờ tải. Đánh đổi: image nặng hơn nhưng khởi động nhanh và chạy được
# cả khi không có mạng — đúng tinh thần "dữ liệu không rời khỏi máy".
RUN python3 -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Cổng Gradio dùng. EXPOSE chỉ là "khai báo cho người đọc" —
# muốn vào được thật vẫn phải có -p 7860:7860 lúc docker run.
EXPOSE 7860

# Gradio mặc định chỉ nghe 127.0.0.1 (chỉ trong container tự nghe mình).
# Phải đổi thành 0.0.0.0 thì bên ngoài container mới vào được.
ENV GRADIO_SERVER_NAME=0.0.0.0

# Lệnh chạy khi container khởi động
CMD ["python3", "src/app.py"]
