# Pipeline đánh giá mức “chuẩn Andrew Tate” của **TopGPT-774M**

> Mục tiêu: đo lường **(i) độ giống phong cách (style), (ii) độ giống nội dung (topic & lập luận), (iii) độ an toàn (toxicity)**
> cho mỗi câu trả lời do TopGPT sinh ra.

---

## 1. Chuẩn bị bộ dữ liệu đánh giá

| Tập | Mô tả | Số mẫu gợi ý |
|-----|-------|--------------|
| **Prompt Set** | 200 prompt chưa dùng trong SFT/DPO, chia 4 chủ đề: “Self-Help”, “Money & Hustle”, “Dating & Gender”, “Random Small-talk”. | 50 prompt / chủ đề |
| **Reference-Tate** | 30 000 câu (tối đa 128 token) trích YouTube‐Rumble transcript, X/Twitter, podcast - *đóng băng* chỉ để làm chuẩn. | 30 k |
| **Reference-Non-Tate** | 30 000 câu cùng chủ đề nhưng của **Jordan Peterson, Joe Rogan, Naval Ravikant, v.v.** | 30 k |
| **Human-Eval Set** | 200 (<2 % tổng) prompt × 2 đáp án (TopGPT & baseline) để gán nhãn thủ công. | 400 item |

> **Tip**: lưu tất cả ở `/data/eval/*.jsonl`, mỗi dòng:  
> `{ "prompt": "...", "response": "...", "model": "TopGPT" }`

---

## 2. Bộ đo (metrics)

| Thuộc tính | Metric | Thư viện |
|------------|--------|----------|
| **Style Score** | `P_Tate = style_clf.proba( resp )[TATE]` (0-1) | `sentence-transformers`, `sklearn` |
| **Embedding Similarity** | `cos_sim( resp , closest_ref_Tate )` | `sentence-transformers` |
| **Log-Perplexity Ratio** | `Δ ppl = ppl_refTate(resp) − ppl_generic(resp)` (âm ⇒ giống Tate) | `transformers` |
| **Keyword Coverage** | % keyword Tate xuất hiện (matrix, hustle, bugatti, etc.) | Python |
| **Toxicity** | `tox = detoxify.predict( resp )` | `detoxify` |
| **Human Likert 1-5** | “Độ giống Andrew Tate?” | Custom interface |

---
