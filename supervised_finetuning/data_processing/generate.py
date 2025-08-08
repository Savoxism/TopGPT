import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from colorama import Fore 

import json
from typing import List 
from pydantic import BaseModel
from openai import OpenAI

from src.prompt import prompt_template
from dotenv import load_dotenv

load_dotenv()
MODEL_ID = "gpt-4.1-mini"

class Record(BaseModel):
    question: str
    answer: str

class Response(BaseModel):
    generated: List[Record]

def llm_call(data: str, num_records: int = 5) -> list:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "user", "content": prompt_template(data, num_records)}
        ],
        response_format={"type": "json_object"},
        stream=True
    )
    
    data_str = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            print(Fore.LIGHTBLUE_EX + delta + Fore.RESET, end="")
            data_str += delta
    
    data_str = data_str.strip()
    
    try:
        result = json.loads(data_str)
        return result.get("generated", [])
    except json.JSONDecodeError as e:
        print(f"\n{Fore.RED}Error parsing JSON: {e}{Fore.RESET}")
        print(f"{Fore.RED}Raw data: {data_str}{Fore.RESET}")
        return []


converter = DocumentConverter()
path = "data/andrew_tate_sft.pdf" # file path
doc = converter.convert(path).document
chunker = HybridChunker()
chunks = list(chunker.chunk(dl_doc=doc))

dataset = {}
print(f"Total chunks: {len(chunks)}")
some_chunks = chunks[:]  
for i, chunk in enumerate(some_chunks): 
    print(Fore.YELLOW + f"Raw Text:\n{chunk.text[:400]}…" + Fore.RESET)
    enriched_text = chunker.contextualize(chunk=chunk)
    print(Fore.LIGHTMAGENTA_EX + f"Contextualized Tex:\n{enriched_text[:400]}…" + Fore.RESET)

    data = llm_call(enriched_text)
    dataset[i] = {"generated": data, 
                    "context": enriched_text}

with open('data/raw_data.json','w') as f: 
    json.dump(dataset, f)