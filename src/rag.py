import sys
from pathlib import Path

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from ingest import cat_theo_dieu, doc_van_ban  # tái dùng hàm chặng 1
import json

DATA_DIR = Path(__file__).parent.parent / "data"
DB_DIR = str(Path(__file__).parent.parent / "chroma_db")

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def _them_vao_kho(kho: chromadb.Collection, chunks: list[dict]):
    """Embed rồi add vào Chroma — id đánh số tiếp theo id đã có, tránh trùng."""
    da_co = kho.count()
    noi_dungs = [c["noi_dung"] for c in chunks]
    vectors = model.encode(noi_dungs, show_progress_bar=True)
    kho.add(
        ids=[f"c{da_co + i}" for i in range(len(chunks))],
        embeddings=[v.tolist() for v in vectors],
        documents=noi_dungs,
        metadatas=[{"file": c["file"], "dieu": c["dieu"]} for c in chunks],
    )


def nap_kho() -> chromadb.Collection:
    """Nạp lại 29 điều khoản gốc (giống chặng 2)."""
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection("hop_dong")
    except Exception:
        pass
    kho = client.create_collection("hop_dong")
    chunks = json.loads((DATA_DIR / "chunks.json").read_text(encoding="utf-8"))
    _them_vao_kho(kho, chunks)
    return kho


def nap_them_file(kho: chromadb.Collection, duong_dan_str: str):
    """Dynamic upload — nạp thêm 1 hợp đồng MỚI vào kho ngay lúc đang
    chạy, không cần sửa code hay chạy lại ingest.py từ đầu. Đây là điểm
    khác biệt lớn nhất so với chặng 1-2: kho không còn "đóng cứng"."""
    duong_dan = Path(duong_dan_str).expanduser()
    if not duong_dan.exists():
        print(f"⚠️  Không tìm thấy file: {duong_dan}")
        return
    van_ban = doc_van_ban(duong_dan)
    chunks_moi = cat_theo_dieu(van_ban, duong_dan.name)
    _them_vao_kho(kho, chunks_moi)
    print(f"✅ Đã nạp thêm {len(chunks_moi)} điều khoản từ {duong_dan.name}")


def tim_ngu_canh(kho: chromadb.Collection, cau_hoi: str, k: int = 3) -> list[dict]:
    """Giống chặng 2 — lấy top-k điều khoản gần nghĩa nhất."""
    q_vec = model.encode([cau_hoi])[0].tolist()
    kq = kho.query(query_embeddings=[q_vec], n_results=k)
    return [
        {"noi_dung": doc, "file": meta["file"], "dieu": meta["dieu"]}
        for doc, meta in zip(kq["documents"][0], kq["metadatas"][0])
    ]


def tao_prompt(cau_hoi: str, ngu_canh: list[dict]) -> str:
    """Prompt 4 phần (Bài 57): VAI TRÒ + NGỮ CẢNH + CÂU HỎI + LUẬT TRẢ LỜI.
    Luật quan trọng nhất: PHẢI trích dẫn nguồn, và PHẢI nói "không biết"
    nếu ngữ cảnh không đủ — cấm bịa (hallucination, LLM tự "sáng tác")."""
    khoi_ngu_canh = "\n\n".join(
        f"[{d['file']} · {d['dieu']}]\n{d['noi_dung']}" for d in ngu_canh
    )
    return f"""Bạn là trợ lý pháp lý, chỉ trả lời DỰA TRÊN các điều khoản dưới đây.

NGỮ CẢNH (các điều khoản liên quan):
{khoi_ngu_canh}

CÂU HỎI: {cau_hoi}

LUẬT BẮT BUỘC:
1. Trả lời ngắn gọn, dễ hiểu, tiếng Việt.
2. Cuối mỗi ý PHẢI ghi nguồn dạng [file · Điều X].
3. Nếu ngữ cảnh KHÔNG chứa thông tin trả lời được câu hỏi, PHẢI trả lời
   đúng câu: "Em không tìm thấy điều khoản nào nói về việc này." — TUYỆT
   ĐỐI không tự bịa câu trả lời ngoài ngữ cảnh trên.
4. CHỈ được dùng đúng những gì CHỮ TRONG NGỮ CẢNH ghi rõ. KHÔNG được tự
   suy luận, tự diễn giải, tự kết luận thay cho ngữ cảnh — kể cả khi câu
   hỏi trông có vẻ liên quan. Nếu ngữ cảnh không nói RÕ RÀNG về đúng tình
   huống được hỏi, PHẢI trả lời như luật 3, không được đoán.
"""


def hoi(kho: chromadb.Collection, cau_hoi: str):
    ngu_canh = tim_ngu_canh(kho, cau_hoi)
    prompt = tao_prompt(cau_hoi, ngu_canh)
    resp = ollama.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": prompt}],
    options={"temperature": 0},
)
    print("\n🤖", resp.message.content)


if __name__ == "__main__":
    kho = nap_kho()

    them = input("\n📎 Đường dẫn hợp đồng muốn nạp thêm (Enter để bỏ qua): ").strip()
    if them:
        nap_them_file(kho, them)

    print("\nGõ câu hỏi để RAG trả lời có trích dẫn (Enter rỗng để thoát):")
    while True:
        cau_hoi = input("\n💬 Hỏi: ").strip()
        if not cau_hoi:
            break
        hoi(kho, cau_hoi)
