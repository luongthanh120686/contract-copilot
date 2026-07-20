# ============================================================
# CHẶNG 5B — ML CỔ ĐIỂN vs LLM: cùng một việc, hai cách làm
# 📖 Ôn lại: Bài 35 (hồi quy logistic) + Bài 37 (đánh giá model,
#            overfitting, cross-validation) — GĐ3 · Bài 63 (eval)
#
# Bài toán: cho 1 điều khoản → nó thuộc LOẠI nào (7 loại)?
#   Cách A — ML cổ điển: TF-IDF (đếm chữ) + LogisticRegression.
#            Máy HỌC từ 29 ví dụ người gán nhãn.
#   Cách B — LLM zero-shot: hỏi thẳng Qwen, KHÔNG dạy ví dụ nào.
# Rồi so: đúng bao nhiêu %, nhanh chậm ra sao, tốn gì.
#
# ĐÂY LÀ MẢNH GĐ3 CUỐI CÙNG — và bài học lớn nhất không phải
# "cái nào giỏi hơn" mà là "KHI NÀO dùng cái nào".
# Chạy: python3 src/classify.py
# ============================================================
import json
import time
from pathlib import Path

import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = Path(__file__).parent.parent / "data"


def nap_du_lieu() -> tuple[list[str], list[str], dict]:
    """Ghép chunks.json (nội dung) với nhan.json (đáp án người gán)."""
    chunks = json.loads((DATA_DIR / "chunks.json").read_text(encoding="utf-8"))
    ho_so = json.loads((DATA_DIR / "nhan.json").read_text(encoding="utf-8"))
    bang_nhan, cac_loai = ho_so["nhan"], ho_so["_cac_loai"]

    X, y = [], []
    for c in chunks:
        khoa = f"{c['file']} · {c['dieu']}"
        if khoa in bang_nhan:          # chỉ lấy điều khoản đã có nhãn
            X.append(c["noi_dung"])
            y.append(bang_nhan[khoa])
    return X, y, cac_loai


# ---------- CÁCH A: ML cổ điển ----------

def chay_ml(X: list[str], y: list[str]) -> tuple[list[str], float]:
    """TF-IDF + LogisticRegression, đánh giá bằng cross-validation.

    TF-IDF (Bài 35): đổi câu chữ thành vector ĐẾM TỪ có đánh trọng số —
    từ hiếm (như "cọc", "phạt") được coi trọng hơn từ đâu cũng có ("và").
    Khác embedding chặng 2: TF-IDF chỉ đếm mặt chữ, KHÔNG hiểu nghĩa —
    "tiền cọc" và "khoản đặt trước" với nó là hai thứ hoàn toàn khác nhau.

    cross_val_predict (Bài 37): chia 3 lượt, mỗi lượt train trên 2/3 và
    đoán 1/3 còn lại → mỗi mẫu đều được đoán bởi model CHƯA từng thấy nó.
    Dữ liệu ít nên chia train/test một lần là quá may rủi.
    """
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    mo_hinh = LogisticRegression(max_iter=1000, class_weight="balanced")

    t0 = time.perf_counter()
    Xv = vec.fit_transform(X)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    du_doan = cross_val_predict(mo_hinh, Xv, y, cv=cv)
    giay = time.perf_counter() - t0
    return list(du_doan), giay


# ---------- CÁCH B: LLM zero-shot ----------

def chay_llm(X: list[str], cac_loai: dict) -> tuple[list[str], float]:
    """Hỏi thẳng Qwen, không đưa ví dụ nào (zero-shot = "không phát bài mẫu").

    Ép JSON + temperature=0 — hai bài học đã học ở chặng 3b và 4.
    Vẫn phải "dọn" đầu ra: LLM có thể trả nhãn lạ ngoài danh sách,
    nên luôn kiểm tra lại thay vì tin thẳng (output LLM = untrusted).
    """
    mo_ta = "\n".join(f"- {k}: {v}" for k, v in cac_loai.items())
    du_doan = []
    t0 = time.perf_counter()
    for i, doan in enumerate(X, 1):
        prompt = (f"Phân loại điều khoản hợp đồng sau vào ĐÚNG MỘT loại:\n{mo_ta}\n\n"
                  f"ĐIỀU KHOẢN:\n{doan}\n\n"
                  'Trả về JSON đúng dạng: {"loai": "tên_loại"} — chỉ dùng tên có trong danh sách.')
        resp = ollama.chat(
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )
        try:
            loai = json.loads(resp.message.content).get("loai", "")
        except json.JSONDecodeError:
            loai = ""
        if loai not in cac_loai:       # nhãn lạ → ghi nhận là sai, không sửa hộ
            loai = f"?{loai[:20]}"
        du_doan.append(loai)
        print(f"   ...LLM đã phân loại {i}/{len(X)}", end="\r")
    giay = time.perf_counter() - t0
    print(" " * 40, end="\r")
    return du_doan, giay


def main():
    X, y, cac_loai = nap_du_lieu()
    print(f"\n📊 {len(X)} điều khoản · {len(set(y))} loại · nhãn do người gán tay")
    print(f"   Phân bố: {json.dumps({k: y.count(k) for k in sorted(set(y))}, ensure_ascii=False)}\n")

    print("🔵 CÁCH A — TF-IDF + LogisticRegression (máy học từ ví dụ)...")
    du_doan_ml, giay_ml = chay_ml(X, y)
    acc_ml = accuracy_score(y, du_doan_ml)

    print("🟢 CÁCH B — Qwen zero-shot (không dạy ví dụ nào)...")
    du_doan_llm, giay_llm = chay_llm(X, cac_loai)
    acc_llm = accuracy_score(y, du_doan_llm)

    print("\n" + "=" * 64)
    print(f"{'':22}{'ML cổ điển':>16}{'LLM zero-shot':>18}")
    print(f"{'Độ chính xác':22}{acc_ml:>15.0%}{acc_llm:>18.0%}")
    print(f"{'Tổng thời gian':22}{giay_ml:>14.2f}s{giay_llm:>17.2f}s")
    print(f"{'Thời gian / điều khoản':22}{giay_ml/len(X)*1000:>13.0f}ms"
          f"{giay_llm/len(X)*1000:>16.0f}ms")
    print(f"{'Cần dữ liệu gán nhãn':22}{'CÓ — 29 mẫu':>16}{'KHÔNG':>18}")
    print(f"{'Chạy được offline':22}{'có':>16}{'có (Ollama)':>18}")
    print("=" * 64)

    print("\n🔍 Những điều khoản CẢ HAI cách cùng đoán sai (khó thật sự):")
    kho = [(x[:60], t, a, b) for x, t, a, b in zip(X, y, du_doan_ml, du_doan_llm)
           if t != a and t != b]
    for x, t, a, b in kho:
        print(f"   • {x}...\n     đúng={t} · ML={a} · LLM={b}")
    if not kho:
        print("   (không có — mỗi cách sai ở chỗ khác nhau)")

    print("\n📋 Chi tiết ML cổ điển sai ở loại nào:")
    print(classification_report(y, du_doan_ml, zero_division=0))


if __name__ == "__main__":
    main()
