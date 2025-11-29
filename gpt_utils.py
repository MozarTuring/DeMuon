# %pip install -r requirements_gpt.txt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator



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

# ---------- dataset ------------------------------------------
class SeqDataset(Dataset):
    """return (input, target) where target is input shifted left"""
    def __init__(self, tokens, block_size):
        self.tokens     = tokens
        self.block_size = block_size
    def __len__(self):
        return len(self.tokens) - self.block_size
    def __getitem__(self, i):
        chunk = self.tokens[i : i+self.block_size+1]
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
    val_ds     = SeqDataset(val_tokens, args.block_size)
    val_loader = DataLoader(val_ds,
                            batch_size=args.eval_batch_size,
                            shuffle=False,
                            generator=g)

    # 4) Prepare data loaders
    loader_ls = []
    rounds_per_epoch = []
    for partid, part in enumerate(partitions):
        ds = SeqDataset(part, args.block_size)
        loader_ls.append(DataLoader(ds, batch_size=args.train_batch_size, shuffle=True,generator=g))
        rounds_per_epoch.append(len(loader_ls[partid]))

    return loader_ls, val_loader, vocab_size, rounds_per_epoch, vocab
