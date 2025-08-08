import os
import sys
import json
import random
import atexit
import signal
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from colorama import Fore, Style, init as colorama_init
from prompt import generation_prompt

# python DPO/src/generate_dpo_data.py
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
    
load_dotenv()
MODEL_ID      = os.getenv("GEN_MODEL_ID", "gpt-4.1-nano")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INPUT_PATH    = "supervised_finetuning/data/instructions.json"
OUTPUT_PATH   = "DPO/data/dpo_train.jsonl"
MAX_EXAMPLES  = 2000         # adjust if needed
FLUSH_EVERY   = 25          # write to disk in batches 

colorama_init(autoreset=True)

def load_instructions(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def chat_json(client: OpenAI, model: str, prompt: str) -> Dict:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        stream=True,
    )
    data_str = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            data_str += delta
    try:
        return json.loads(data_str.strip())
    except json.JSONDecodeError:
        print(f"{Fore.YELLOW}⚠️  JSON parse failure – received:\n{data_str}{Style.RESET_ALL}")
        return {}

def generate_answer(client: OpenAI, question: str) -> str:
    prompt = generation_prompt(question, num_candidates=1)
    obj    = chat_json(client, MODEL_ID, prompt)
    return obj.get("candidates", [{}])[0].get("text", "").strip()

def flush_buffer(path: str, buf: List[str]):
    if not buf:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for line in buf:
            f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"{Fore.GREEN}Flushed {len(buf)} records → {path}{Style.RESET_ALL}")
    buf.clear()


# ------------------------------------------------------------------------ #
client   = OpenAI(api_key=OPENAI_API_KEY)
dataset  = load_instructions(INPUT_PATH)

# adjust to resume from existing data
already_written = 0
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        already_written = sum(1 for _ in f)

print(f"{Fore.CYAN}Found {already_written} existing records – resuming…{Style.RESET_ALL}")

dataset = dataset[already_written:]  

remaining_budget = MAX_EXAMPLES - already_written
total = min(len(dataset), remaining_budget)
buffer   : List[str] = []

# graceful exit / flush handlers
def _flush_and_exit(*_):
    print(f"\n{Fore.YELLOW}Exiting – flushing {len(buffer)} remaining records...{Style.RESET_ALL}")
    flush_buffer(OUTPUT_PATH, buffer)
    sys.exit(0)

atexit.register(_flush_and_exit)
signal.signal(signal.SIGINT, _flush_and_exit)

for i in tqdm(range(total), desc=Fore.CYAN + "Building DPO dataset" + Style.RESET_ALL, ncols=100):
    item = dataset[i]
    question = item.get("question", "").strip()
    original_answer = item.get("answer", "").strip()

    if not question or not original_answer:
        continue

    new_answer = generate_answer(client, question)
    if not new_answer:
        continue

    record = {
        "prompt" : question,
        "chosen" : new_answer,       # preferred (model-generated)
        "rejected": original_answer  # baseline from dataset
    }
    buffer.append(json.dumps(record, ensure_ascii=False))

    if len(buffer) >= FLUSH_EVERY:
        flush_buffer(OUTPUT_PATH, buffer)

flush_buffer(OUTPUT_PATH, buffer)
print(f"{Fore.CYAN}Done. Wrote {total} examples to {OUTPUT_PATH}{Style.RESET_ALL}")

