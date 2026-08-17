# train a small bpe/subword shakespeare model (tiktoken / gpt2)
# tuned for tokenized (tiktoken) dataset rather than character-level

out_dir = 'out-shakespeare-word'
eval_interval = 250
eval_iters = 200
log_interval = 10

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-word'
wandb_run_name = 'mini-gpt-word'

# use the tiktoken BPE-prepared dataset
dataset = 'shakespeare_word'

gradient_accumulation_steps = 1
batch_size = 32
block_size = 64  # context of up to 64 tokens

# model
n_layer = 8
n_head = 8
n_embd = 512

dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
