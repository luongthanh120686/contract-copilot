# ============================================================
# CHẶNG 5 — EVAL: chấm điểm hệ thống bằng BỘ CÂU HỎI VÀNG
# 📖 Ôn lại: Bài 63 (Eval & an toàn LLM) + Bài 37 GĐ3 (đánh giá model)
# Vì sao cần: "chạy thử 1 câu thấy ổn" KHÔNG chứng minh hệ thống tốt.
# Anh đã tự chứng minh điều đó ở chặng 3 (hỏi 5 lần ra 5 câu khác nhau).
# Eval = biến việc thử tay đó thành PHÉP ĐO LẶP LẠI ĐƯỢC.
#
# 3 CHỈ SỐ ta đo (mỗi cái bắt một loại lỗi khác nhau):
#   1. retrieval@3 — kho vector có LẤY ĐÚNG điều khoản lên không?
#      (đo riêng phần TÌM — lỗi ở đây thì LLM giỏi mấy cũng chịu)
#   2. citation    — câu trả lời có trích ĐÚNG nguồn không?
#      (bắt đúng lỗi "bịa Điều 5" anh tìm ra ở chặng 4)
#   3. refusal     — câu ngoài phạm vi có TỪ CHỐI không, hay bịa?
#      (an toàn LLM — Bài 63)
# Chạy: python3 src/eval.py
# ============================================================
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rag import hoi_lay_chuoi, nap_kho, tim_ngu_canh

DATA_DIR = Path(__file__).parent.parent / "data"
CAU_TU_CHOI = "không tìm thấy"  # dấu hiệu model chịu nói "em không biết"


def nap_golden() -> list[dict]:
    """Bộ câu hỏi vàng: câu hỏi + đáp án đúng do NGƯỜI soạn tay.
    Đây là 'thước đo' — sai thước thì mọi con số phía sau vô nghĩa,
    nên từng câu phải đối chiếu với hợp đồng thật trước khi tin."""
    return json.loads((DATA_DIR / "golden.json").read_text(encoding="utf-8"))


def cham_retrieval(kho, cau: dict, k: int = 3) -> bool | None:
    """Chỉ số 1 — kho vector có kéo đúng điều khoản vào top-k không?
    Trả None cho câu bẫy (không có đáp án đúng thì không chấm được)."""
    if cau["phai_tu_choi"]:
        return None
    ngu_canh = tim_ngu_canh(kho, cau["cau_hoi"], k=k)
    lay_duoc = {f"{d['file']} · {d['dieu']}" for d in ngu_canh}
    return any(nguon in lay_duoc for nguon in cau["nguon_dung"])


def cham_citation(cau: dict, tra_loi: str) -> bool | None:
    """Chỉ số 2 — câu trả lời có nhắc đúng file + Điều cần trích không?"""
    if cau["phai_tu_choi"]:
        return None
    return any(
        nguon.split(" · ")[0] in tra_loi and nguon.split(" · ")[1] in tra_loi
        for nguon in cau["nguon_dung"]
    )


def cham_noi_dung(cau: dict, tra_loi: str) -> bool | None:
    """Chỉ số 2b — nội dung có chứa CON SỐ / Ý CHÍNH bắt buộc không?
    (trích đúng nguồn nhưng nói sai số vẫn là sai)"""
    if cau["phai_tu_choi"]:
        return None
    thap = tra_loi.lower()
    return all(tu.lower() in thap for tu in cau["tu_khoa_bat_buoc"])


def cham_refusal(cau: dict, tra_loi: str) -> bool | None:
    """Chỉ số 3 — câu bẫy có được TỪ CHỐI đúng không?
    Trả None cho câu thường (câu thường mà từ chối là lỗi khác, đã bắt
    ở chỉ số citation/nội dung rồi)."""
    if not cau["phai_tu_choi"]:
        return None
    return CAU_TU_CHOI in tra_loi.lower()


def ty_le(ket: list[bool | None]) -> str:
    co_cham = [x for x in ket if x is not None]
    if not co_cham:
        return "—"
    dung = sum(co_cham)
    return f"{dung}/{len(co_cham)} = {dung / len(co_cham) * 100:.0f}%"


def main():
    kho = nap_kho()
    golden = nap_golden()
    print(f"\n📋 Chấm {len(golden)} câu hỏi vàng — mỗi câu gọi Qwen 1 lần...\n")

    r_ret, r_cit, r_noi, r_ref = [], [], [], []
    for i, cau in enumerate(golden, 1):
        tra_loi = hoi_lay_chuoi(kho, cau["cau_hoi"])

        ret = cham_retrieval(kho, cau)
        cit = cham_citation(cau, tra_loi)
        noi = cham_noi_dung(cau, tra_loi)
        ref = cham_refusal(cau, tra_loi)
        r_ret.append(ret); r_cit.append(cit); r_noi.append(noi); r_ref.append(ref)

        def dau(x):
            return "—" if x is None else ("✅" if x else "❌")

        loai = "🪤 BẪY" if cau["phai_tu_choi"] else "     "
        print(f"{i}. {loai} {cau['cau_hoi']}")
        print(f"   tìm {dau(ret)} · trích {dau(cit)} · nội dung {dau(noi)} · từ chối {dau(ref)}")
        if False in (ret, cit, noi, ref):  # chỉ in chi tiết khi CÓ lỗi
            print(f"   → máy trả lời: {tra_loi[:160]}...")
        print()

    print("=" * 58)
    print(f"1. retrieval@3 (tìm đúng điều khoản): {ty_le(r_ret)}")
    print(f"2. citation    (trích đúng nguồn)   : {ty_le(r_cit)}")
    print(f"2b. nội dung   (đúng số/ý chính)    : {ty_le(r_noi)}")
    print(f"3. refusal     (từ chối câu bẫy)    : {ty_le(r_ref)}")
    print("=" * 58)
    print("Ghi lại 4 con số này. Mỗi lần sửa prompt/model/chunking, chạy lại")
    print("để biết mình đang TIẾN hay LÙI — thay vì đoán bằng cảm giác.")


if __name__ == "__main__":
    main()
