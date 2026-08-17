"""
Prepare the Shakespeare dataset using tiktoken (GPT-2 BPE).
Saves train.bin, val.bin containing the token ids, and meta.pkl containing the
encoder and decoder and vocab_size.
"""
import os
import requests
import tiktoken
import numpy as np
import pickle

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# split train/val
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken (gpt2 BPE)
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# choose dtype depending on vocab size
vocab_size = enc.n_vocab if hasattr(enc, 'n_vocab') else 50257
if vocab_size < 65536:
    dtype = np.uint16
else:
    dtype = np.uint32

# export to bin files
train_arr = np.array(train_ids, dtype=dtype)
val_arr = np.array(val_ids, dtype=dtype)
train_arr.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_arr.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save meta
meta = {
    'vocab_size': vocab_size,
    'encoding': 'gpt2',
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
