# ============================================================
# CHẶNG 2 — Trí nhớ ngữ nghĩa: embed từng điều khoản → Chroma →
#           tìm theo NGHĨA (chưa cần LLM!)
# 📖 Ôn lại: Bài 51 (embedding = tọa độ GPS của ý nghĩa)
#            Bài 52 (vector database, Top-K)
#            Bài 20 GĐ2 (vector) + Bài 26 (cosine similarity)
# Chạy: python3 src/search.py  → gõ câu hỏi, xem top-3 điều khoản.
# ============================================================
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data"
DB_DIR = str(Path(__file__).parent.parent / "chroma_db")

# Model embedding ĐA NGÔN NGỮ (hiểu tiếng Việt) — tải 1 lần rồi cache.
# Cùng một model phải dùng cho CẢ lúc lưu lẫn lúc hỏi (bẫy Bài 54!).
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def nap_kho() -> chromadb.Collection:
    """Đọc chunks.json → embed từng điều khoản → cất vào Chroma."""
    client = chromadb.PersistentClient(path=DB_DIR)  # lưu xuống ổ đĩa
    # Xóa collection cũ để nạp lại sạch (bẫy Bài 54: dữ liệu cũ-mới lẫn lộn)
    try:
        client.delete_collection("hop_dong")
    except Exception:
        pass
    kho = client.create_collection("hop_dong")

    chunks = json.loads((DATA_DIR / "chunks.json").read_text(encoding="utf-8"))
    noi_dungs = [c["noi_dung"] for c in chunks]
    vectors = model.encode(noi_dungs, show_progress_bar=True)  # chữ → tọa độ

    kho.add(
        ids=[f"c{i}" for i in range(len(chunks))],
        embeddings=[v.tolist() for v in vectors],
        documents=noi_dungs,
        metadatas=[{"file": c["file"], "dieu": c["dieu"]} for c in chunks],
    )
    print(f"Đã nạp {len(chunks)} điều khoản vào kho vector.")
    return kho


def tim(kho: chromadb.Collection, cau_hoi: str, k: int = 3):
    """Embed câu hỏi bằng CÙNG model → lấy top-k điều khoản gần nghĩa."""
    q_vec = model.encode([cau_hoi])[0].tolist()
    kq = kho.query(query_embeddings=[q_vec], n_results=k)
    for doc, meta, dist in zip(
        kq["documents"][0], kq["metadatas"][0], kq["distances"][0]
    ):
        # distance càng NHỎ càng giống (bẫy Bài 52 — ngược với similarity!)
        print(f"\n--- [{meta['file']} · {meta['dieu']}] (distance {dist:.3f})")
        print(doc[:220], "...")


if __name__ == "__main__":
    kho = nap_kho()
    print("\nGõ câu hỏi để tìm điều khoản (Enter rỗng để thoát):")
    while True:
        cau_hoi = input("\n🔎 Hỏi: ").strip()
        if not cau_hoi:
            break
        tim(kho, cau_hoi)
