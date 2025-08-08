import json
from colorama import Fore

instructions = []

file_path = "data/raw_data.json"
with open(file_path, "r") as f:
    data = json.load(f)

    for key, chunk in data.items():
        for pairs in chunk['generated']:
            question = pairs['question']
            answer = pairs['answer'] if 'answer' in pairs else pairs['Answer']  # Handle both 'answer' and 'Answer'

            context_pair = {
                'question': f"{question}",
                'answer': f"{answer}",
            }
            instructions.append(context_pair)
        print(Fore.YELLOW + str(chunk)) 
        print('\n~~~~~~~~~~~~~~~~~~~~~')

with open('data/instructions.json','w') as f: 
    json.dump(instructions,f) 

with open('data/instructions.json','r') as f: 
    data = json.load(f)
    print(Fore.LIGHTMAGENTA_EX + str(data[:10]))