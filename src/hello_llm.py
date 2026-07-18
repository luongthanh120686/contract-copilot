
# ============================================================
# CHẶNG 0 — Lời chào đầu tiên tới LLM chạy NGAY TRÊN MÁY ANH
# 📖 Ôn lại: Bài 45 (LLM là gì) + Bài 47 (gọi LLM)
# Khác giáo trình: dùng Ollama (miễn phí, local) thay vì API trả phí.
# Không cần API key — model qwen2.5:7b đang nằm trong máy.
# ============================================================
import ollama  # thư viện nói chuyện với Ollama server trên máy

cau_hoi = "Chào bạn! Hãy giải thích RAG trong 2 câu tiếng Việt thật dễ hiểu."

# Gửi hội thoại cho model — giống hệt cấu trúc messages của bài 47,
# chỉ khác: chạy trên chính Mac này, không tốn đồng nào, không cần mạng.
resp = ollama.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": cau_hoi}],
)

print(resp.message.content)
