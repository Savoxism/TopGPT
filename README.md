# TopGPT
Creating a persona with Transformers

# Lộ trình hoàn chỉnh huấn luyện **TopGPT-774M**  
> Kiến trúc: **GPT-2 774 M tham số** (decoder-only, mask-LM)  
> Các giai đoạn: **Pre-training → Continued Pre-training (CPT) → Supervised Fine-Tuning (SFT) → Direct Preference Optimization (DPO)**.  
> Tất cả con số đều đã thu hẹp cho quy mô 774 M tham số.

---

## 0. Cấu hình mô hình & tokenizer

| Mục | Thông số |
|-----|----------|
| Kiến trúc | GPT-2 (12 layer, 64 head, d_model = 1 536) |
| Tham số | ≈ 774 M |
| Kích thước embedding | 50 k BPE (sử dụng **tiktoken** hoặc **SentencePiece**)<br>✓ Có thể tái dùng vocab GPT-2 gốc để giảm thời gian |
| Chuẩn hoá | LayerNorm trước attention & MLP (pre-LN) |
| Kỹ thuật tăng tốc | FlashAttention-2, RMS prop weight decay, mixed-precision (FP16/bfloat16) |

---

## 1. Pre-training nền (general LM)

### 1.a Dữ liệu  
- **Tổng token**: **≈ 100 B** (gấp ~12 lần tham số ⇒ tỷ lệ 130 token/param).  
- **Thành phần**  
  - **Common Crawl** (Slim-Pajama, RefinedWeb) ≈ 60 B
  - **Wikipedia** (en + vi) ≈ 3 B  
  - **Books3 + Project Gutenberg** ≈ 5 B  
  - **Stack Exchange, arXiv, HackerNews** ≈ 8 B  
  - **News/Reddit dedupe** ≈ 24 B  
- **Tiền xử lý**:  
  1. Deduplicate (Exact-hash + MinHash).  
  2. Bộ lọc profanity, ngôn ngữ ≠ English/Vietnamese.  
  3. Chunk 2048 token, lưu **WebDataset tar**.

### 1.b Phần cứng & thời gian ước tính  
| Hạng mục | Số lượng/Giá trị |
|----------|------------------|
| GPU | **16 × A100 80 GB** (hoặc 8 × H100 80 GB) |
| Batch global | 2 M token (32 GPU-batch × 64 seq × 1 k token/seq) |
| LR schedule | 1 e-4 cosine, warmup 2 % |
| Bước cập nhật | 50 M step (≈ 1 epoch trên 100 B token) |
| Thời gian | ~10 ngày ( ≈ 3.2 PFLOP; nhờ Flash-Attn) |
| Lưu trữ | 200 GB nén raw, 1 TB shard tar |

### 1.c Thư viện/Stack  
- **PyTorch ≥ 2.2**, **Transformers ≥ 0.23**  
- **DeepSpeed (ZeRO-3)** hoặc **Megatron-LM**  
- **flash-attn-2**, **bitsandbytes**  
- **webdataset**, **datasets**  
- Giám sát: **wandb**, **tqdm** để hiển thị tiến độ.

---

## 2. Continued Pre-training (CPT) – “Andrew Tate tone”

### 2.a Dữ liệu chuyên hoá  
| Nguồn | Thu thập | Token ước tính |
|-------|----------|----------------|
| Video/phỏng vấn (YouTube, Rumble) | `yt-dlp` + Whisper-X | 20 M |
| Podcast transcript | RSS crawl | 8 M |
| Twitter/X, Instagram | API + Scraper | 5 M |
| Blog/newsletter cá nhân | Wayback Machine | 5 M |
| **Tổng** |  | **≈ 40 M** |

> Con số này ~ 0 .04 × token pre-train ⇒ vừa đủ “nhiễm” phong cách mà không **mode-collapse**.

### 2.b Phần cứng  
- **4 × A100 80 GB** hoặc **4 × RTX 4090**.  
- Huấn luyện 2 epoch qua 40 M token → 2–3 giờ.

### 2.c Thư viện  
- **LoRA/QLoRA** (`peft`) nếu cần tiết kiệm VRAM.  
- **Accelerate** single-node.  
- **sentencepiece --sample`** (BPE-dropout) để tăng robust.

---

## 3. Supervised Fine-Tuning (SFT)

### 3.a Bộ dữ liệu SFT  
| Loại | Mục tiêu | # mẫu |
|------|----------|-------|
| Instruction → Response | Hỏi đáp ngắn phong cách Tate | 15 k |
| Multi-turn chat | 4–6 lượt/đoạn | 4 k |
| Rewrite/Style transfer | Chuyển văn bản “trung lập → Tate” | 3 k |
| **Tổng** |  | **≈ 22 k** (~ 7 M token) |

- Sinh data bằng gemini / gemma / gpt 4.1 mini+ prompt “Bạn là Andrew Tate…”, sau đó **human spot-check** 20 %.

### 3.b Phần cứng & thiết lập  
- **1 × A100 40 GB** (QLoRA-4bit) hoặc **2 × RTX 3090**.  
- LR = 2 e-5, batch = 256 seq, 3 epoch → ≈ 2 giờ.

### 3.c Thư viện  
- **trl.SFTTrainer**  
- **bitsandbytes==0.43**, **peft**  
- Callbacks `tqdm`, **wandb**.

---

## 4. Direct Preference Optimization (DPO)

### 4.a Data pairwise  
| Nguồn tạo | # cặp | Phương pháp |
|-----------|-------|-------------|
| ChatGPT self-play | 35 k | Prompt 2 đáp án → phân loại “Tate hơn” |
| Human ranking (crowd) | 7 k | 3 vote/cặp |
| Toxicity adversarial | 3 k | prompt jailbreak / protest |
| **Tổng** | **45 k cặp** (~ 6 M token) |

### 4.b Phần cứng  
- **4 × A100 40 GB**; batch = 512 seq, β = 0.1.  
- Thời gian ≈ 4 giờ.

### 4.c Thư viện  
- **trl.DPOTrainer**  
- **flash-attn-2**  
- **wandb sweep** (lr, β).

---

## 5. Đánh giá & triển khai

| Hạng mục | Tool |
|----------|------|
| Chat eval | `lm-eval-harness`, `MT-Bench` |
| Toxicity & jailbreak | `mjt-eval`, `harm-bench` |
| An toàn (tùy mục tiêu) | `safe-completion`, *reward-model-detox* |
| Phục vụ | **vLLM** + PagedAttention |
| Quant hoá | **AWQ** / **GPTQ** (4 bit) |
| Giám sát runtime | Prometheus + Grafana |

---

## 6. Danh sách package Python gợi ý

```txt
torch==2.2.*
transformers==0.23.*
accelerate>=0.30
datasets>=2.19
bitsandbytes>=0.43
peft>=0.10
trl>=0.8
flash-attn>=2.5
deepspeed==0.14
sentencepiece
tiktoken
webdataset
yt-dlp
whisperx
wandb
tqdm
vllm
