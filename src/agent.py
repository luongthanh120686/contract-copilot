# ============================================================
# CHẶNG 4 — Agent: Qwen TỰ QUYẾT ĐỊNH dùng công cụ nào, mấy lần
# 📖 Ôn lại: Bài 58 (AI Agent là gì) + Bài 59 (Tool calling)
#            + Bài 60 (vòng lặp ReAct — ta TỰ VIẾT TAY, không framework)
# Khác chặng 3: RAG là đường ống CỨNG (luôn tìm → luôn trả lời).
# Agent là vòng lặp MỀM: model tự chọn "bước kế tiếp làm gì" —
# tìm điều khoản? bấm máy tính? hay đủ thông tin rồi, trả lời?
# Ta thấy được từng bước suy nghĩ của nó in ra màn hình.
# Chạy: python3 src/agent.py
# ============================================================
import json
import re
import sys 
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).parent))
from rag import nap_kho, tim_ngu_canh # tái dùng kho vector chặng 2-3

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------- 3 CÔNG CỤ (tools) — hàm Python thường, LLM không tự chạy được ----------

def tim_dieu_khoan(kho, cau_hoi: str) -> str:
    """Công cụ 1: tìm 3 điều khoản gần nghĩa nhất (tái dùng chặng 2)."""
    ngu_canh = tim_ngu_canh(kho, cau_hoi)
    return "\n\n".join(
        f"[{d['file']} · {d['dieu']}]\n{d['noi_dung']}" for d in ngu_canh
    ) 


def tinh_tien_phat(so_tien: float, phan_tram_moi_ngay: float, so_ngay: float) -> str:
    """Công cụ 2: máy tính tiền phạt. LLM rất DỞ tính nhẩm (Bài 59 —
    lý do số 1 phải đưa máy tính cho agent), nên phép nhân giao cho Python."""
    phat = so_tien * (phan_tram_moi_ngay / 100) * so_ngay
    return (f"{so_tien:,.0f} đồng × {phan_tram_moi_ngay}%/ngày × {so_ngay:.0f} ngày "
            f"= {phat:,.0f} đồng")


def doc_nguyen_van(ten_file: str, dieu: str) -> str:
    """Công cụ 3: đọc NGUYÊN VĂN 1 điều khoản từ chunks.json —
    để kiểm chứng trích dẫn, tránh model nhớ mang máng rồi diễn lại sai."""
    chunks = json.loads((DATA_DIR / "chunks.json").read_text(encoding="utf-8"))
    for c in chunks:
        if c["file"] == ten_file and c["dieu"] == dieu:
            return c["noi_dung"]
    return f"Không tìm thấy {dieu} trong {ten_file}."


# ---------- BỘ NÃO: prompt dạy Qwen luật chơi của agent ----------

LUAT_AGENT = """Bạn là agent trợ lý hợp đồng. Bạn KHÔNG tự biết nội dung hợp đồng \
và KHÔNG được tính nhẩm — phải dùng công cụ.

CÔNG CỤ CÓ SẴN:
1. tim_dieu_khoan — tham_so: {"cau_hoi": "câu cần tìm"} — tìm 3 điều khoản liên quan nhất.
2. tinh_tien_phat — tham_so: {"so_tien": số, "phan_tram_moi_ngay": số, "so_ngay": số} — máy tính tiền phạt.
3. doc_nguyen_van — tham_so: {"ten_file": "tên file", "dieu": "Điều N"} — đọc nguyên văn 1 điều khoản.

LUẬT BẮT BUỘC — mỗi lượt CHỈ in ra đúng MỘT khối JSON, không chữ nào khác:
- Muốn dùng công cụ: {"suy_nghi": "vì sao cần", "cong_cu": "tên", "tham_so": {...}}
- Đủ thông tin rồi:  {"suy_nghi": "vì sao đủ", "tra_loi": "câu trả lời cuối, kèm nguồn [file · Điều X]"}
BƯỚC ĐẦU TIÊN luôn luôn là tim_dieu_khoan — mức phạt, số phần trăm, điều kiện \
đều phải LẤY TỪ KẾT QUẢ CÔNG CỤ, cấm lấy từ trí nhớ dù trông hiển nhiên. \
Mọi phép tính PHẢI qua tinh_tien_phat. Nguồn trong tra_loi phải là tên file \
và số Điều THẬT đã thấy trong kết quả công cụ (ví dụ [hop-dong-thue-nha.txt · Điều 8]), \
cấm ghi chung chung kiểu [file · Điều X]."""


def goi_llm(messages: list[dict]) -> str:
    resp = ollama.chat(
        model="qwen2.5:7b",
        messages=messages,
        format="json",               # ÉP model in JSON hợp lệ 100% (constrained
                                     # decoding — chặn từ gốc lỗi ngoặc kép lồng nhau)
        options={"temperature": 0},  # bài học chặng 3b: nhất quán
    )
    return resp.message.content


def tach_json(chu: str) -> dict:
    """Model đôi khi bọc JSON trong ```...``` — cắt lấy từ { đầu tới } cuối.
    Save = untrusted input thì output LLM cũng vậy: luôn phòng thủ khi parse."""
    dau, cuoi = chu.find("{"), chu.rfind("}")
    if dau == -1 or cuoi == -1:
        raise ValueError(f"Không thấy JSON trong: {chu[:80]}")
    return json.loads(chu[dau : cuoi + 1])


def chay_agent(kho, cau_hoi: str, toi_da: int = 6):
    """Vòng lặp ReAct viết tay (Bài 60): Nghĩ → Hành động → Quan sát → lặp."""
    messages = [
        {"role": "system", "content": LUAT_AGENT},
        {"role": "user", "content": cau_hoi},
    ]
    da_goi = set()  # chống agent lặp vô hạn cùng 1 lệnh (bẫy Bài 60)
    for buoc in range(1, toi_da + 1):
        try:
            quyet_dinh = tach_json(goi_llm(messages))
        except (ValueError, json.JSONDecodeError) as e:
            print(f"⚠️  Bước {buoc}: model in JSON hỏng ({e}) — nhắc nó làm lại.")
            messages.append({"role": "user", "content": "Sai định dạng. CHỈ in một khối JSON đúng luật."})
            continue

        print(f"\n🧠 Bước {buoc} — suy nghĩ: {quyet_dinh.get('suy_nghi', '(không ghi)')}")

        if "tra_loi" in quyet_dinh:                      # agent tự thấy đủ → dừng
            print(f"\n✅ TRẢ LỜI: {quyet_dinh['tra_loi']}")
            return
        ten = quyet_dinh.get("cong_cu")
        ts = quyet_dinh.get("tham_so", {})
        chu_ky = f"{ten}|{json.dumps(ts, ensure_ascii=False, sort_keys=True)}"
        if chu_ky in da_goi:  # gọi lại y hệt lệnh cũ → nhắc nó tiến lên
            messages.append({"role": "assistant", "content": json.dumps(quyet_dinh, ensure_ascii=False)})
            messages.append({"role": "user", "content":
                "Bạn vừa lặp lại đúng công cụ + tham số cũ — kết quả không đổi. "
                "Dùng thông tin ĐÃ CÓ: gọi tinh_tien_phat nếu cần tính, hoặc chốt tra_loi."})
            print(f"🔁 Bước {buoc}: agent định gọi lại lệnh cũ ({ten}) — đã chặn, nhắc nó tiến lên.")
            continue
        da_goi.add(chu_ky)
        print(f"🔧 Gọi công cụ: {ten}({ts})")
        try:
            if ten == "tim_dieu_khoan":
                ket_qua = tim_dieu_khoan(kho, str(ts.get("cau_hoi", "")))
            elif ten == "tinh_tien_phat":
                ket_qua = tinh_tien_phat(float(ts["so_tien"]),
                                         float(ts["phan_tram_moi_ngay"]),
                                         float(ts["so_ngay"]))
            elif ten == "doc_nguyen_van":
                ket_qua = doc_nguyen_van(str(ts.get("ten_file", "")), str(ts.get("dieu", "")))
            else:
                ket_qua = f"Không có công cụ tên '{ten}'."
        except (KeyError, TypeError, ValueError) as e:
            ket_qua = f"Tham số sai: {e}"
        print(f"👁  Kết quả: {ket_qua[:200]}{'...' if len(ket_qua) > 200 else ''}")

        # Ghi lại lượt này vào hội thoại để model "quan sát" rồi nghĩ tiếp
        messages.append({"role": "assistant", "content": json.dumps(quyet_dinh, ensure_ascii=False)})
        messages.append({"role": "user", "content": f"KẾT QUẢ CÔNG CỤ:\n{ket_qua}"})

    print("\n⛔ Hết lượt mà chưa chốt được câu trả lời — tăng toi_da hoặc xem lại luật.")


if __name__ == "__main__":
    kho = nap_kho()
    print("\nAgent sẵn sàng — hỏi câu cần TÌM + TÍNH, ví dụ:")
    print('  "Tôi thuê nhà 12 triệu/tháng, chậm thanh toán 20 ngày thì bị phạt bao nhiêu tiền?"')
    while True:
        cau_hoi = input("\n💬 Hỏi (Enter rỗng để thoát): ").strip()
        if not cau_hoi:
            break
        chay_agent(kho, cau_hoi)

