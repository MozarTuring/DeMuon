# %pip install -r requirements_gpt.txt

import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchtext_compat import Multi30k, get_tokenizer, build_vocab_from_iterator
import itertools



from utils import *

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layer=2, n_head=4, max_len=128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)                                
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks  = nn.ModuleList(
            nn.TransformerDecoderLayer(d_model, n_head, dim_feedforward=4*d_model, batch_first=True)
            for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)                       # (B,T,C)
        x   = tok + self.pos_emb[:, :T, :]
        # generate subsequent mask once
        mask = torch.triu(torch.ones(T, T, device=idx.device), 1).bool()
        for block in self.blocks:
            # x is used for both self‑attn (query) and key/value
            x = block(x, x, tgt_mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# ---------- FineWeb binary data loading (matches train_gpt_tiny.py format) ------

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def fineweb_data_generator(files: list, seq_len: int, device: str = "cuda"):
    """Yields non-overlapping (input, target) pairs of length seq_len from binary shards."""
    file_iter = itertools.cycle(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + seq_len + 1 > len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos : pos + seq_len + 1]
        inputs = buf[:-1].to(device=device, dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device=device, dtype=torch.int64, non_blocking=True)
        pos += seq_len + 1
        yield inputs, targets


def get_loaders_fineweb(args, device):
    """Set up data generators for FineWeb binary shards.
    Returns (train_generators, val_generator, vocab_size, rounds_per_epoch)."""
    import glob

    train_files = sorted(Path(f) for f in glob.glob(args.train_files))
    val_files = sorted(Path(f) for f in glob.glob(args.val_files))
    assert len(train_files) > 0, f"No train files found matching {args.train_files}"
    assert len(val_files) > 0, f"No val files found matching {args.val_files}"

    jwp(f"FineWeb train shards: {len(train_files)}, val shards: {len(val_files)}")

    # Partition train files among workers (round-robin)
    worker_files = [[] for _ in range(args.n_workers)]
    for i, f in enumerate(train_files):
        worker_files[i % args.n_workers].append(f)

    for i in range(args.n_workers):
        jwp(f"  Worker {i}: {len(worker_files[i])} train shards")

    # Create per-worker training generators
    train_generators = []
    for wid in range(args.n_workers):
        gen = fineweb_data_generator(worker_files[wid], args.train_seq_len, device)
        train_generators.append(gen)

    # Validation generator
    val_gen = fineweb_data_generator(val_files, args.val_seq_len, device)

    # Estimate rounds per epoch (tokens per worker / seq_len)
    # Each shard is ~100M tokens; approximate
    first_shard_tokens = int(torch.from_file(str(train_files[0]), False, 256, dtype=torch.int32)[2])
    tokens_per_worker = first_shard_tokens * len(worker_files[0])
    rounds_per_epoch = tokens_per_worker // (args.train_seq_len + 1)
    jwp(f"  Estimated tokens per worker: ~{tokens_per_worker:,}")
    jwp(f"  Estimated rounds per epoch: ~{rounds_per_epoch:,}")

    vocab_size = args.vocab_size
    return train_generators, val_gen, vocab_size, rounds_per_epoch


# ---------- Multi30k dataset (legacy) ------------------------------------------
class SeqDataset(Dataset):
    """Non-overlapping contiguous chunks (like train_gpt.py's data generator).
    Returns (input, target) where target is input shifted by 1."""
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_chunks = len(tokens) // (seq_len + 1)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, i):
        start = i * (self.seq_len + 1)
        chunk = self.tokens[start : start + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return x, y

def yield_tokens(split, tokenizer):
    for eng, _de in Multi30k(split=split, language_pair=("en", "de")):
        yield tokenizer(eng.lower())



def get_loaders(args):
    # 1) Build vocabulary from Multi30k train
    tok = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(yield_tokens("train", tok), specials=["<pad>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    vocab_size = len(vocab)
    jwp(f"Vocab size = {vocab_size}")

    # 2) Tokenise entire train split into one flat list (quick & dirty)
    tokens = []
    val_tokens = []
    for eng, _de in Multi30k(split="train", language_pair=("en","de")):
        tokens.extend(vocab(tok(eng.lower())))
        # vocab maps token to integer and note that it's extend rather than append
    for eng, _ in Multi30k(split="valid", language_pair=("en","de")):   # ← official dev set
        val_tokens.extend(vocab(tok(eng.lower())))
    random.shuffle(tokens)

    # 3) Partition tokens equally among workers
    jwp(f"total train tokens = {len(tokens)}")
    part_len = len(tokens) // args.n_workers
    jwp(f"each worker has {part_len} train tokens")
    partitions = [tokens[i*part_len:(i+1)*part_len] for i in range(args.n_workers)]

    jwp(f"total val tokens = {len(val_tokens)}")
    val_ds     = SeqDataset(val_tokens, args.val_seq_len)
    jwp(f"val chunks = {len(val_ds)} (non-overlapping, seq_len={args.val_seq_len})")
    val_loader = DataLoader(val_ds,
                            batch_size=args.eval_batch_size,
                            shuffle=False,
                            generator=g)

    # 4) Prepare data loaders
    loader_ls = []
    rounds_per_epoch = []
    for partid, part in enumerate(partitions):
        ds = SeqDataset(part, args.train_seq_len)
        loader_ls.append(DataLoader(ds, batch_size=args.train_batch_size, shuffle=True,generator=g))
        rounds_per_epoch.append(len(loader_ls[partid]))
    jwp(f"train chunks per worker = {len(loader_ls[0].dataset)} (non-overlapping, seq_len={args.train_seq_len})")

    return loader_ls, val_loader, vocab_size, rounds_per_epoch, vocab
