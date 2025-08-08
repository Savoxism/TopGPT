import json
import os
from pydantic import BaseModel
from colorama import Fore
from openai import OpenAI
from dotenv import load_dotenv
from src.prompt import verification_prompt_template
from tqdm import tqdm 

load_dotenv()
MODEL_ID = "gpt-4.1-nano"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Score(BaseModel):
    score: int
    explanation: str

class Rank(BaseModel):
    accuracy: Score
    style: Score

def llm_call(record: str) -> dict:

    prompt = verification_prompt_template(record=record)
                
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        stream=True
    )
    
    data_str = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            print(Fore.LIGHTBLUE_EX + delta + Fore.RESET, end="")
            data_str += delta
    
    try:
        return json.loads(data_str.strip())
    except json.JSONDecodeError as e:
        print(f"\n{Fore.RED}Error parsing JSON: {e}{Fore.RESET}")
        print(f"{Fore.RED}Raw response: {data_str}{Fore.RESET}")
        # Return default low scores if parsing fails
        return {
            "accuracy": {"score": 1, "explanation": "Failed to parse model response"},
            "style": {"score": 1, "explanation": "Failed to parse model response"}
        }

def append_jsonl(filename, data_list):
    with open(filename, "a", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

quality = []
instructions = []
print(f"{Fore.GREEN}Starting Andrew Tate style verification using {MODEL_ID}{Fore.RESET}")

try:
    with open("data/instructions.json", "r") as f:
        data = json.load(f)

        total = len(data)
        print(f"{Fore.CYAN}Found {total} Q&A pairs to verify{Fore.RESET}")

        # Start from the 200th data point (index 199, but want 200th onwards, so index 200)
        start_idx = 500
        if start_idx >= total:
            print(f"{Fore.YELLOW}Start index {start_idx} is beyond data length. Nothing to process.{Fore.RESET}")
        else:
            progress_bar = tqdm(enumerate(data[start_idx:], start=start_idx), total=total-start_idx, desc="Verifying records", unit="record")

            for i, pair in progress_bar:
                record_str = f"Question: {pair['question']}\nAnswer: {pair['answer']}"
                progress_bar.set_description(f"Processing record {i+1}/{total}")

                print(f"\n{Fore.YELLOW}Processing record {i+1}/{total}:{Fore.RESET}")
                print(f"{Fore.WHITE}{record_str[:100]}...{Fore.RESET}")

                result = llm_call(record_str)
                accuracy_score = result['accuracy']['score']
                style_score = result['style']['score']

                print(f"\n{Fore.MAGENTA}Scores - Accuracy: {accuracy_score}/10, Style: {style_score}/10{Fore.RESET}")

                if accuracy_score > 6 and style_score > 6:
                    instructions.append(pair)
                    quality.append({**pair, 'quality': result})
                    print(f"{Fore.GREEN}✓ Record accepted (High quality Andrew Tate style){Fore.RESET}")
                    progress_bar.set_postfix({"Accepted": len(instructions), "Rejected": i+1-len(instructions)})
                else:
                    quality.append({**pair, 'quality': result})
                    print(f"{Fore.RED}✗ Record rejected (Low quality - Accuracy: {accuracy_score}, Style: {style_score}){Fore.RESET}")
                    progress_bar.set_postfix({"Accepted": len(instructions), "Rejected": i+1-len(instructions)})

                # Save every 100 records (append, not overwrite)
                if (i + 1) % 100 == 0:
                    append_jsonl('data/verified_instructions.jsonl', instructions)
                    append_jsonl('data/verification_analysis.jsonl', quality)
                    instructions.clear()
                    quality.clear()
                    print(f"{Fore.GREEN}Appended 100 records to .jsonl files{Fore.RESET}")

            # Save any remaining records at the end
            if instructions:
                append_jsonl('data/verified_instructions.jsonl', instructions)
            if quality:
                append_jsonl('data/verification_analysis.jsonl', quality)
            print(f"{Fore.GREEN}Appended remaining records to .jsonl files{Fore.RESET}")

except FileNotFoundError:
    print(f"{Fore.RED}Error: Could not find data/instruction.json{Fore.RESET}")
except Exception as e:
    print(f"{Fore.RED}Error during verification process: {str(e)}{Fore.RESET}")