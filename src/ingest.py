# ============================================================
# CHẶNG 1 — Cho Copilot "ăn" hợp đồng: đọc file → cắt theo Điều
# 📖 Ôn lại: Bài 54 (pipeline RAG: Load + Chunk) + Bài 55 (chiến
#            lược cắt đoạn THEO CẤU TRÚC — xịn hơn cắt cố định)
# Kết quả: data/chunks.json — mỗi chunk là MỘT ĐIỀU KHOẢN, kèm
# metadata (file nào, điều số mấy) để sau này trích dẫn nguồn.
# ============================================================
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"   # thư mục chứa hợp đồng
OUT_FILE = DATA_DIR / "chunks.json"                 # nơi lưu kết quả


def doc_van_ban(duong_dan: Path) -> str:
    """Đọc 1 file hợp đồng thành chuỗi chữ. (Bài 13 GĐ1: with open + utf-8)"""
    with open(duong_dan, "r", encoding="utf-8") as f:
        return f.read()


def cat_theo_dieu(van_ban: str, ten_file: str) -> list[dict]:
    """Cắt hợp đồng thành từng ĐIỀU (chunking theo cấu trúc — Bài 55).

    Regex tìm 'Điều <số>.' và cắt từ đó tới ngay trước Điều kế tiếp,
    nên mỗi chunk trọn vẹn một ý — không bị đứt ngang câu như cắt
    theo số ký tự cố định.
    """
    mau = re.compile(r"(Điều \d+\..*?)(?=\nĐiều \d+\.|\Z)", re.DOTALL)
    chunks = []
    for khop in mau.finditer(van_ban):
        noi_dung = khop.group(1).strip()
        so_dieu = re.match(r"Điều (\d+)", noi_dung).group(1)
        chunks.append({
            "file": ten_file,           # nguồn: hợp đồng nào
            "dieu": f"Điều {so_dieu}",  # nguồn: điều số mấy
            "noi_dung": noi_dung,       # ruột của chunk
        })
    return chunks


def main():
    tat_ca = []
    for duong_dan in sorted(DATA_DIR.glob("*.txt")):
        van_ban = doc_van_ban(duong_dan)
        chunks = cat_theo_dieu(van_ban, duong_dan.name)
        print(f"  {duong_dan.name}: {len(chunks)} điều khoản")
        tat_ca.extend(chunks)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(tat_ca, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {len(tat_ca)} chunk vào {OUT_FILE.name}")


if __name__ == "__main__":
    main()
