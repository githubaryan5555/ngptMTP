"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from configurator import *  # load config overrides if run as script

# default sampling config
init_from = globals().get('init_from', 'resume')
out_dir = globals().get('out_dir', 'out')
start = globals().get('start', "\n")
num_samples = globals().get('num_samples', 10)
max_new_tokens = globals().get('max_new_tokens', 500)
temperature = globals().get('temperature', 0.8)
top_k = globals().get('top_k', 200)
seed = globals().get('seed', 1337)
device = globals().get('device', 'cuda')
dtype = globals().get('dtype', 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16')
compile_model = globals().get('compile', False)


def load_checkpoint_model(init_from, out_dir, device):
    """Load model from checkpoint or pretrained GPT-2."""
    if init_from == 'resume':
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        # build model from saved model_args
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model_args = checkpoint.get('model_args', {})
        meta = checkpoint.get('config', {})
        return model, model_args, meta
    elif init_from.startswith('gpt2'):
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))
        return model, getattr(model, 'config', {}), {}
    else:
        raise ValueError(f"Unknown init_from: {init_from}")


def get_encoder_decoder(meta, out_dir):
    # prefer dataset meta in checkpoint/config if present
    # meta from data files: contains 'stoi' and 'itos' for char datasets
    if meta and 'dataset' in meta:
        meta_path = os.path.join('data', meta['dataset'], 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                ds_meta = pickle.load(f)
            if 'stoi' in ds_meta and 'itos' in ds_meta:
                stoi, itos = ds_meta['stoi'], ds_meta['itos']
                encode = lambda s: [stoi[c] for c in s]
                decode = lambda l: ''.join([itos[i] for i in l])
                return encode, decode
            if 'encoding' in ds_meta and ds_meta['encoding'] == 'gpt2':
                enc = tiktoken.get_encoding('gpt2')
                encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
                decode = lambda l: enc.decode(l)
                return encode, decode
    # fallback to assuming gpt2 encoding
    enc = tiktoken.get_encoding('gpt2')
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    return encode, decode


def generate_samples(model, encode, decode, start, num_samples, max_new_tokens, temperature=1.0, top_k=None, device='cuda'):
    model.eval()
    model.to(device)
    # prepare prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')


def main():
    torch.manual_seed(seed)
    if 'cuda' in device:
        torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_actual = device
    model, model_args, meta = load_checkpoint_model(init_from, out_dir, device_actual)
    # print multitok info
    mtp = None
    if hasattr(model, 'config') and getattr(model.config, 'multitok_pred', None) is not None:
        mtp = model.config.multitok_pred
    elif model_args and 'multitok_pred' in model_args:
        mtp = model_args['multitok_pred']
    print(f"Model multitok_pred = {mtp}")

    encode, decode = get_encoder_decoder(meta, out_dir)

    if compile_model:
        model = torch.compile(model)

    generate_samples(model, encode, decode, start, num_samples, max_new_tokens, temperature, top_k, device_actual)


if __name__ == '__main__':
    main()
