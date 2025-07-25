import os
import requests
import tiktoken
import numpy as np 

input_file_path = os.path.join(os.path.dirname(__file__), 'shakespeare.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

n = len(data)
train_data = data[: int(0.9 * n)]
val_data = data[int(n * 0.9):]

# Encoding with the tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

print(f"train data has {len(train_ids)} tokens")
print(f"val data has {len(val_ids)} tokens")

# Save to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
# np.save(os.path.join(os.path.dirname(__file__), 'train.bin'), arr=train_ids)
# np.save(os.path.join(os.path.dirname(__file__), 'val.bin'), arr=val_ids) 

