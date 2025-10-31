# %pip install -r requirements_gpt.txt
# exit()
# in kaggle notebook, !pip may not be pip attached to the current python environment
# sometimes need to restart the kernel after pip install

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator



from utils import *

jwp(f"{torch.__version__},{np.__version__}") # check whether pip install succeeded, if not restart the kernel and re‑run

# ---------- decoder‑only Transformer ---------------------



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

@torch.no_grad()
def eval_loss(model, loader, loss_fn):
    model.eval()
    tot, ntok = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        n      = y.numel()
        tot   += loss.item()*n
        ntok  += n
    return tot/ntok


# ---------- main ----------------------------------------------
def main():
    # parameters defaulted setting
    parser = comoon_args()
    parser.add_argument("--block_size",type=int, default=64)
    parser.add_argument("--d_model",   type=int, default=256)
    parser.add_argument("--n_layer",   type=int, default=2)
    parser.add_argument("--n_head",    type=int, default=4)
    parser.add_argument("--max_len",   type=int, default=128)

    args = parser.parse_args(['--n_workers','1','--train_batch_size','128','--eval_batch_size','512','--network','','--alg','cenmuon','--epochs','12', '--log_interval','600'])

    # args = parser.parse_args(['--n_workers','8','--train_batch_size','64','--eval_batch_size','512','--network','ring','--alg','muon','--epochs','12', '--log_interval','100'])

    # parameters recording and monitoring
    wandb.init(project="my-kaggle-project", name=f'gpt_{args.alg}_{args.network}')
    artifact = wandb.Artifact("my_model", type="model")
    
    os.makedirs("graphs", exist_ok=True)
    os.makedirs(f"output/{args.network}", exist_ok=True)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

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
                            shuffle=False)

    # 4) Prepare data loaders
    loaders = []
    rounds_per_epoch = []
    for partid, part in enumerate(partitions):
        ds = SeqDataset(part, args.block_size)
        loaders.append(DataLoader(ds, batch_size=args.train_batch_size, shuffle=True))
        rounds_per_epoch.append(len(loaders[partid]))
    iters = [iter(loader) for loader in loaders]
    max_round_per_epoch = max(rounds_per_epoch) # to finish one epoch for all workers, we need to run the longest loader for max_round_per_epoch steps
    jwp(f"max_round_per_epoch = {max_round_per_epoch}") 

    # 5) Instantiate models & per‑worker momentum buffers (no built‑in optim)
    
    if args.network == "ring":
        weights = connected_cycle_weights(filename=f"graphs/ring_{args.n_workers}.npy", n=args.n_workers, degree=1)
        mixing = torch.from_numpy(weights).float().to(device)
        # ring: 0.804737854124365
    elif args.network == "exp":
        weights = exponential_graph_weights(filename=f"graphs/exp_{args.n_workers}.npy", n=args.n_workers)
        mixing = torch.from_numpy(weights).float().to(device)
        # exp:  0.5999999999999998 
    elif args.network == "complete":
        weights = complete_graph_weights(filename=f"graphs/complete_{args.n_workers}.npy", n=args.n_workers)
        mixing = torch.from_numpy(weights).float().to(device)
        # complete: 
    
    workers = []
    base_model = MiniGPT(vocab_size, args.d_model, args.n_layer, args.n_head, args.max_len)
    for _ in range(args.n_workers):
        m = MiniGPT(vocab_size, args.d_model, args.n_layer, args.n_head, args.max_len)
        m.load_state_dict(base_model.state_dict())
        # jwp(m.tok_emb(torch.tensor([0])))
        m.to(device)
        workers.append(m)
    del base_model
    # return "done"
    
    if args.alg == 'dsgd':
        args.lr=1e-2
        lr = args.lr
    elif args.alg == 'dsgd_gclip_decay':
        args.lr=10
        args.l2_clip_bd=0.1
        lr, l2_clip_bd = args.lr, args.l2_clip_bd
    elif args.alg == 'gt_dsgd':
        lr = args.lr
        y_list, g_prev_list = [], []
        for _ in range(args.n_workers):
            # use last m in the memory
            y_list.append({name: torch.zeros_like(param) for name, param in m.named_parameters()})
            g_prev_list.append({name: torch.zeros_like(param) for name, param in m.named_parameters()})
    elif args.alg in ["gt_nsgdm", "muon", 'cenmuon']:
        args.lr=1e-1
        args.mom=0.8
        lr, mom = args.lr, args.mom
        y_list, m_list = [], []
        for _ in range(args.n_workers):
            # use last m in the memory
            y_list.append({name: torch.zeros_like(param) for name, param in m.named_parameters()})
            m_list.append({name: torch.zeros_like(param) for name, param in m.named_parameters()})
    elif args.alg == 'sen':
        lr, mom, phi, tau = args.lr, args.mom, args.phi, args.tau
        m_list = []
        for _ in range(args.n_workers):
            m_list.append({name: torch.zeros_like(param) for name, param in m.named_parameters()})
    total_params = sum(p.numel() for p in workers[0].parameters())
    jwp(f"total_params = {total_params}")
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    # 6) Example mixing matrix: uniform average
    # W = torch.full((args.n_workers, args.n_workers), 1/args.n_workers, device="cuda")

    # 7) CSV logging
    header = ["round"]
    header += [f"w{i}_val"   for i in range(args.n_workers)]
    header += [f"w{i}_train" for i in range(args.n_workers)]
    loss_table = [header]

    total_rounds = args.epochs * max_round_per_epoch
    for r in range(1, total_rounds+1):
        if r == 1:
            val_losses = [eval_loss(m, val_loader, loss_fn) for m in workers]
            jwp(f"Initial validation losses: " +
                    ", ".join(f"{l:.4f}" for l in val_losses))
        round_losses = []
        # one local step per worker
        for wid, model in enumerate(workers):
            try:
                x, y = next(iters[wid])
            except StopIteration:          # start a new local epoch
                iters[wid] = iter(loaders[wid]) # reset the loader
                x, y = next(iters[wid])

            x, y = x.to(device), y.to(device)

            # ----- forward/backward -----
            model.train()
            logits = model(x)
            loss   = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            model.zero_grad(set_to_none=True)
            loss.backward()

            # use local gradient to update local buffers
            if args.alg == 'dsgd':
                with torch.no_grad():
                    # update buffers block by block
                    for (name, p) in model.named_parameters():
                        if p.grad is None:
                            continue
                        g = p.grad
                        p.data -= lr * g
            elif args.alg == 'dsgd_gclip_decay':
                # clip gradient norm globally
                lr = args.lr / r
                l2_clip_bd = args.l2_clip_bd * r**0.4
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=l2_clip_bd)
                # update buffers block by block
                with torch.no_grad():
                    for (name, p) in model.named_parameters():
                        if p.grad is None:
                            continue
                        g = p.grad
                        p.data -= lr * g 
            elif args.alg == 'gt_dsgd':
                with torch.no_grad():
                    # update buffers block by block
                    for (name, p) in model.named_parameters():
                        if p.grad is None:
                            continue
                        g = p.grad
                        g_prev = g_prev_list[wid][name]
                        y = y_list[wid][name]
                        # buffer update
                        y.add_(g).add_(g_prev, alpha=-1.0)
                        y_list[wid][name] = y
                        g_prev_list[wid][name] = g
            elif args.alg in ["gt_nsgdm","muon",'cenmuon']: 
                with torch.no_grad():
                    # update buffers block by block
                    for (name, p) in model.named_parameters():
                        if p.grad is None:
                            continue
                        g = p.grad
                        m = m_list[wid][name]
                        y = y_list[wid][name]
                        # buffer update
                        m_temp = m.mul(mom).add(g, alpha=1-mom) # get v^t
                        y.add_(m_temp).add_(m, alpha=-1.0)
                        y_list[wid][name] = y
                        m_list[wid][name] = m_temp 
            elif args.alg == 'sen':
                with torch.no_grad():
                    # update buffers block by block
                    for (name, p) in model.named_parameters():
                        if p.grad is None:
                            continue
                        g = p.grad
                        m = m_list[wid][name]
                        # buffer update
                        temp = sclip(g.add(m, alpha=-1.0), phi, r, tau)
                        m.mul_(mom/r**0.5).add_(temp, alpha=1-mom/r**0.5)
                        m_list[wid][name] = m
                        p.data -= lr/r**0.2 * m
            round_losses.append(loss.item())

        # ---- mixing after every local step, i.e., communication after every local step  -----
        if args.alg == 'dsgd' or args.alg == 'dsgd_gclip_decay':
            mix_params(workers, mixing)
        elif args.alg == "gt_dsgd":
            y_list = mix_y_list(y_list, mixing)
            # use y_list to update parameters on each worker
            for wid, model in enumerate(workers):
                with torch.no_grad():
                    # update parameters block by block
                    for (name, p) in model.named_parameters():
                        if p.grad is None:
                            continue
                        p.data -= lr * y_list[wid][name]
            mix_params(workers, mixing)
        elif args.alg in ["gt_nsgdm", "muon", 'cenmuon']:
            # mix y_list
            if args.alg in ['gt_nsgdm','muon']:
                y_list = mix_y_list(y_list, mixing)
            # use normalized y_list to update parameters on each worker
            svd_time=0.0
            for wid, model in enumerate(workers):
                # normalize model's y globally
                if args.alg == "gt_nsgdm":
                    normalized_y = normalize_tensor_dict(y_list[wid])
                elif args.alg in ["muon", 'cenmuon']:
                    normalized_y = dict()
                    for name in y_list[wid]:
                        tmp=y_list[wid][name].squeeze()
                        if tmp.ndim==1 or name in ["tok_emb","pos_emg"]:
                            normalized_y[name] = tmp/torch.norm(tmp)
                        elif tmp.ndim==2:
                            tmp_start = time.time()
                            U, S, Vt = torch.linalg.svd(tmp, full_matrices=False)
                            svd_time+=time.time()-tmp_start
                            normalized_y[name] = U @ Vt
                        else:
                            jwp(f"Error: not implemented for {name} {tmp.shape} ndim>2")
                            return "error"
                with torch.no_grad():
                    # update parameters block by block
                    for (name, p) in model.named_parameters():
                        if p.grad is None:
                            continue
                        # if name == "pos_emb":
                        #     jwp(p.data.shape,normalized_y[name].shape)
                        #     return None
                        p.data -= lr * normalized_y[name]
            if args.alg in ['gt_nsgdm','muon']:
                mix_params(workers, mixing)
        elif args.alg == 'sen':
            mix_params(workers, mixing)
       
        # logging
        # jwp(f"svd_time={svd_time:.4f}")
        if r % args.log_interval == 0 or r == total_rounds or r == 1:
            val_losses = [eval_loss(m, val_loader, loss_fn) for m in workers]
            loss_table.append([r] + val_losses + round_losses)
            jwp(f"Round {r}/{total_rounds}: {round_losses},{val_losses}")
            
    # after the for‑round loop
    end(args=args, artifact=artifact, loss_table=loss_table)

if __name__ == "__main__":
    main()

# P100 slower than T4
