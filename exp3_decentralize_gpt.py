from __future__ import annotations
import json
import random
import statistics
import time
import math
from functools import lru_cache

from torch.utils.data import DataLoader, TensorDataset

import utils as _utils_mod
import gpt_utils as _gpt_utils_mod
from gpt_utils import *
from utils import *
from gpt60m import GPT as GPT60M, next_multiple_of_n

SUPPORTED_ALGS = ["demuon", "dsgd", "dsgd_gclip_decay", "gt_dsgd", "gt_nsgdm", "sen"]


def quick2json(inp_path, inp_data):
    with open(inp_path, "w", encoding="utf8") as wf:
        wf.write(json.dumps(inp_data, ensure_ascii=False, indent=2))


@torch.no_grad()
def eval_loss(model, val_gen, sliding_window_num_blocks, val_tokens, val_seq_len):
    """Evaluate model on val_tokens total tokens from the val generator."""
    model.eval()
    val_steps = val_tokens // val_seq_len
    tot = 0.0
    for step in range(val_steps):
        inputs, targets = next(val_gen)
        loss = model(inputs, targets, sliding_window_num_blocks)
        tot += loss.item()
    return tot / val_steps


def run_single_seed(args, seed, csv_path=None):
    """Run a full training loop for one seed. Returns the loss_table and
    final metrics dict.  If csv_path is given, the CSV is flushed to disk
    after every validation evaluation."""

    set_random_seed(seed)
    new_g = torch.Generator()
    new_g.manual_seed(seed)
    _utils_mod.g = new_g
    _gpt_utils_mod.g = new_g

    header = ["round"]
    header += [f"w{i}_val" for i in range(args.n_workers)]
    header += [f"w{i}_train" for i in range(args.n_workers)]
    header += [f"w{i}_val_ppl" for i in range(args.n_workers)]
    header += [
        "avg_val_loss",
        "avg_val_ppl",
        "consensus_err",
        "comm_rounds",
        "cumul_time_sec",
        "iter_time_sec",
    ]
    loss_table = [header]

    loader_ls, val_gen, vocab_size, rounds_per_epoch_est = get_loaders_fineweb(args, device)
    jwp(f"[seed={seed}] estimated rounds_per_epoch={rounds_per_epoch_est}")

    # Dynamic sliding window schedule (matches train_gpt_tiny.py)
    @lru_cache(maxsize=256)
    def get_window_size_blocks(step: int, num_iterations: int):
        x = step / num_iterations
        x = max(0.0, min(x, 1.0))
        factor = 4 * x ** 3 - 6 * x ** 2 + 3 * x
        window_size = next_multiple_of_n(3456 * factor, n=128)
        return torch.tensor(window_size // 128, dtype=torch.int32, device=device)

    if args.epochs is not None:
        total_rounds = args.epochs * rounds_per_epoch_est
        jwp(f"[seed={seed}] Using --epochs={args.epochs}: total_rounds={total_rounds} "
            f"({args.epochs} * {rounds_per_epoch_est})")
    else:
        total_rounds = args.num_iterations

    mixing, _ = get_graph(args, device)

    # --- LR schedule: stable then decay (matches train_gpt_tiny.py) ---
    def get_lr(step: int):
        x = step / total_rounds
        x = max(0.0, min(x, 1.0))
        if x < 1 - args.cooldown_frac:
            return 1.0
        else:
            return (1 - x) / args.cooldown_frac

    # --- Create models ---
    model_ls = list()
    for i in range(args.n_workers):
        model = GPT60M(
            vocab_size, num_layers=args.n_layer, num_heads=args.n_head,
            model_dim=args.d_model, max_seq_len=args.max_len
        )
        model.float()
        if len(model_ls) > 0:
            model.load_state_dict(model_ls[0].state_dict())
        model.to(device)
        model_ls.append(model)
    model_ls = [torch.compile(m, dynamic=False) for m in model_ls]
    jwp(f"[seed={seed}] All {args.n_workers} models created and torch.compiled (dynamic=False)")

    ref_model = model_ls[-1]

    # --- Parameter grouping (matches train_gpt_tiny.py) ---
    # hidden_matrix_params: 2D params from model.blocks (for Muon-style update)
    # embed_params: embedding layers
    # scalar_params: 1D lambda/skip parameters
    # head_params: lm_head_w
    def classify_params(model):
        hidden_matrix_names = set()
        embed_names = set()
        scalar_names = set()
        head_names = set()
        for name, p in model.named_parameters():
            if "lm_head_w" in name:
                head_names.add(name)
            elif "embed" in name or "value_embeds" in name:
                embed_names.add(name)
            elif "scalars" in name or "lambdas" in name or "skip_weights" in name:
                scalar_names.add(name)
            elif p.ndim >= 2 and ("blocks" in name or "block" in name):
                hidden_matrix_names.add(name)
            elif p.ndim >= 2:
                hidden_matrix_names.add(name)
            else:
                scalar_names.add(name)
        return hidden_matrix_names, embed_names, scalar_names, head_names

    hidden_matrix_names, embed_names, scalar_names, head_names = classify_params(ref_model)
    jwp(f"[seed={seed}] Parameter groups: hidden_matrix={len(hidden_matrix_names)}, "
        f"embed={len(embed_names)}, scalar={len(scalar_names)}, head={len(head_names)}")

    # --- Create AdamW optimizers for non-hidden params (only for demuon) ---
    adam_optimizers = []
    if args.alg == "demuon":
        for wid, model in enumerate(model_ls):
            adam_params = []
            for name, p in model.named_parameters():
                if name in head_names:
                    adam_params.append(dict(params=[p], lr=1/320, initial_lr=1/320))
                elif name in embed_names:
                    adam_params.append(dict(params=[p], lr=0.3, initial_lr=0.3))
                elif name in scalar_names:
                    adam_params.append(dict(params=[p], lr=0.015, initial_lr=0.015))
            opt = torch.optim.AdamW(adam_params, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0)
            adam_optimizers.append(opt)

    # --- algorithm-specific buffer init (only for hidden_matrix_params for demuon) ---
    alg = args.alg
    y_list, m_list, g_prev_list = [], [], []

    if alg == "demuon":
        for _ in range(args.n_workers):
            y_list.append(
                {n: torch.zeros_like(p) for n, p in ref_model.named_parameters()
                 if n in hidden_matrix_names}
            )
            m_list.append(
                {n: torch.zeros_like(p) for n, p in ref_model.named_parameters()
                 if n in hidden_matrix_names}
            )
    elif alg == "gt_nsgdm":
        for _ in range(args.n_workers):
            y_list.append(
                {n: torch.zeros_like(p) for n, p in ref_model.named_parameters()}
            )
            m_list.append(
                {n: torch.zeros_like(p) for n, p in ref_model.named_parameters()}
            )
    elif alg == "gt_dsgd":
        for _ in range(args.n_workers):
            y_list.append(
                {n: torch.zeros_like(p) for n, p in ref_model.named_parameters()}
            )
            g_prev_list.append(
                {n: torch.zeros_like(p) for n, p in ref_model.named_parameters()}
            )
    elif alg == "sen":
        for _ in range(args.n_workers):
            m_list.append(
                {n: torch.zeros_like(p) for n, p in ref_model.named_parameters()}
            )

    # --- communication cost estimate ---
    use_msgn = alg == "demuon" and args.msgn != 0
    bytes_per_round = communication_bytes_per_round(
        model_ls[0], args.n_workers, use_msgn
    )
    jwp(f"[seed={seed}] Estimated comm bytes/round: {bytes_per_round:,}")

    iteration_times = []
    cumul_time = 0.0
    comm_rounds_count = 0

    # log initial state (round 0): forward pass on first batch (no param update)
    jwp(f"[seed={seed}] Starting round 0 train-loss forward passes for {args.n_workers} workers...")
    jwp(f"[seed={seed}] NOTE: first forward pass will be slow due to torch.compile warmup")
    round0_train_losses = []
    for wid, model in enumerate(model_ls):
        inputs, targets = next(loader_ls[wid])
        model.eval()
        with torch.no_grad():
            t0 = time.perf_counter()
            loss = model(inputs, targets, get_window_size_blocks(0, total_rounds))
            dt = time.perf_counter() - t0
            jwp(f"[seed={seed}] Worker {wid} round0 forward done in {dt:.2f}s, loss={loss.item():.4f}")
        round0_train_losses.append(loss.item())

    jwp(f"[seed={seed}] Starting round 0 validation eval for {args.n_workers} workers...")
    val_losses_0 = []
    for vi, m in enumerate(model_ls):
        t0 = time.perf_counter()
        vl = eval_loss(m, val_gen, get_window_size_blocks(0, total_rounds), args.val_tokens, args.val_seq_len)
        dt = time.perf_counter() - t0
        jwp(f"[seed={seed}] Worker {vi} eval_loss done in {dt:.2f}s, val_loss={vl:.4f}")
        val_losses_0.append(vl)
    val_ppls_0 = [math.exp(vl) for vl in val_losses_0]
    avg_val_0 = statistics.mean(val_losses_0)
    avg_ppl_0 = math.exp(avg_val_0)
    cons_err_0 = consensus_error(model_ls)
    row_0 = (
        [0]
        + val_losses_0
        + round0_train_losses
        + val_ppls_0
        + [round(avg_val_0, 6), round(avg_ppl_0, 4), round(cons_err_0, 6), 0, 0.0, 0.0]
    )
    loss_table.append(row_0)
    if csv_path is not None:
        with open(csv_path, "w", newline="") as _cf:
            csv.writer(_cf).writerows(loss_table)
    jwp(
        f"[seed={seed}] Round 0 (init): train_loss={[round(l, 4) for l in round0_train_losses]}, "
        f"avg_val={avg_val_0:.4f}, ppl={avg_ppl_0:.2f}, cons_err={cons_err_0:.6f}"
    )

    jwp(f"[seed={seed}] Starting training loop: total_rounds={total_rounds}, n_workers={args.n_workers}")
    jwp(f"[seed={seed}] Effective batch per round: {args.n_workers * args.train_batch_size} sequences "
        f"({args.n_workers} workers x {args.train_batch_size} accum steps)")
    for r in range(1, total_rounds + 1):
        t_start = time.perf_counter()
        round_losses = []

        # ===== LR schedule (stable then decay, matches train_gpt_tiny.py) =====
        lr_mult = get_lr(r)
        muon_lr = args.muon_lr * lr_mult

        # Momentum warmup for demuon hidden_matrix params (0.85 -> 0.95 over 300 steps)
        frac = min(r / 300, 1.0)
        muon_mom = (1 - frac) * 0.85 + frac * args.mom

        # Update AdamW LR schedule (demuon only)
        if adam_optimizers:
            for wid in range(args.n_workers):
                for group in adam_optimizers[wid].param_groups:
                    group["lr"] = group["initial_lr"] * lr_mult

        # ===== per-worker forward/backward + local buffer update =====
        for wid, model in enumerate(model_ls):
            model.train()
            model.zero_grad(set_to_none=True)
            total_loss = torch.tensor(0.0, device=device)
            for _ in range(args.train_batch_size):
                inputs, targets = next(loader_ls[wid])
                step_loss = model(inputs, targets, get_window_size_blocks(r, total_rounds)) / args.train_batch_size
                step_loss.backward()
                total_loss = total_loss + step_loss.detach()
            round_losses.append(total_loss.item())
            if r <= 3:
                jwp(f"[seed={seed}] Round {r} worker {wid} fwd+bwd done, loss={total_loss.item():.4f}")

            with torch.no_grad():
                if alg == "demuon":
                    # Momentum tracking only for hidden_matrix_params
                    for name, p in model.named_parameters():
                        if p.grad is None or name not in hidden_matrix_names:
                            continue
                        m_temp = muon_mom * m_list[wid][name] + (1 - muon_mom) * p.grad
                        y_list[wid][name] = (
                            y_list[wid][name] + m_temp - m_list[wid][name]
                        )
                        m_list[wid][name] = m_temp

                    # AdamW step for non-hidden params
                    adam_optimizers[wid].step()

                elif alg == "dsgd":
                    tmp_lr = args.lr * lr_mult
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        p.data -= tmp_lr * p.grad

                elif alg == "dsgd_gclip_decay":
                    tmp_lr = args.lr * lr_mult
                    cur_clip = args.l2_clip_bd * r**0.4
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=cur_clip
                    )
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        p.data -= tmp_lr * p.grad

                elif alg == "gt_dsgd":
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        g = p.grad
                        y_list[wid][name].add_(g).add_(
                            g_prev_list[wid][name], alpha=-1.0
                        )
                        g_prev_list[wid][name] = g.clone()

                elif alg == "gt_nsgdm":
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        g = p.grad
                        m_temp = m_list[wid][name].mul(args.mom).add(g, alpha=1 - args.mom)
                        y_list[wid][name].add_(m_temp).add_(
                            m_list[wid][name], alpha=-1.0
                        )
                        m_list[wid][name] = m_temp

                elif alg == "sen":
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        g = p.grad
                        m = m_list[wid][name]
                        temp = sclip(g.add(m, alpha=-1.0), args.phi, r, args.tau)
                        m.mul_(args.mom / r**0.5).add_(temp, alpha=1 - args.mom / r**0.5)
                        m_list[wid][name] = m
                        p.data -= args.lr / r**0.2 * m

        # ===== mixing / communication =====
        if alg == "demuon":
            if args.n_workers > 1:
                for _ in range(args.gossip_rounds):
                    y_list = mix_y_list(y_list, mixing)
                    comm_rounds_count += 1

            # Muon-style update for hidden_matrix_params
            for wid, model in enumerate(model_ls):
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if name not in hidden_matrix_names:
                            continue
                        y = y_list[wid][name]
                        tmp_shape = y.shape
                        tmp = y.squeeze()
                        if tmp.ndim < 2:
                            continue

                        if args.msgn == 0:
                            update = y
                        elif args.msgn == 1:
                            # Nesterov-like NS input (matches Muon):
                            # NS(mom * buf + (1-mom) * grad)
                            # For n_workers=1: y = m (the buffer), p.grad is the local grad
                            # For n_workers>1: y is mixed, use plain y as DeMuon design
                            if args.n_workers == 1 and p.grad is not None:
                                ns_input = muon_mom * tmp + (1 - muon_mom) * p.grad.squeeze()
                            else:
                                ns_input = tmp
                            update = zeropower_via_newtonschulz5(
                                ns_input, steps=args.ns_steps
                            ).reshape(tmp_shape)
                        elif args.msgn == 2:
                            U, S, Vt = torch.linalg.svd(tmp, full_matrices=False)
                            update = (U @ Vt).reshape(tmp_shape)

                        # Aspect-ratio LR scaling (matches Muon)
                        aspect_ratio = max(1, p.size(-2) / p.size(-1)) ** 0.5
                        eff_lr = muon_lr * aspect_ratio

                        # Weight decay (LR-coupled, matches Muon)
                        wd_mul = getattr(p, "wd_mul", 1.0)
                        eff_wd = muon_lr * args.muon_wd * wd_mul
                        p.data.mul_(1 - eff_wd)

                        # Apply NS update
                        p.data.add_(update, alpha=-eff_lr)

            if args.n_workers > 1:
                for _ in range(args.gossip_rounds):
                    mix_params(model_ls, mixing)
                    comm_rounds_count += 1

        elif alg in ("dsgd", "dsgd_gclip_decay", "sen"):
            if args.n_workers > 1:
                mix_params(model_ls, mixing)
                comm_rounds_count += 1

        elif alg == "gt_dsgd":
            if args.n_workers > 1:
                y_list = mix_y_list(y_list, mixing)
                comm_rounds_count += 1
            tmp_lr = args.lr * lr_mult
            for wid, model in enumerate(model_ls):
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        p.data -= tmp_lr * y_list[wid][name]
            if args.n_workers > 1:
                mix_params(model_ls, mixing)
                comm_rounds_count += 1

        elif alg == "gt_nsgdm":
            if args.n_workers > 1:
                y_list = mix_y_list(y_list, mixing)
                comm_rounds_count += 1
            tmp_lr = args.lr * lr_mult
            for wid, model in enumerate(model_ls):
                normalized_y = normalize_tensor_dict(y_list[wid])
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        p.data -= tmp_lr * normalized_y[name]
            if args.n_workers > 1:
                mix_params(model_ls, mixing)
                comm_rounds_count += 1

        # ===== logging =====
        t_elapsed = time.perf_counter() - t_start
        iteration_times.append(t_elapsed)
        cumul_time += t_elapsed

        if r <= 3:
            jwp(f"[seed={seed}] Round {r} compute+comm done in {t_elapsed:.2f}s")

        if r % args.log_interval == 0 or r == total_rounds or r == 1:
            jwp(f"[seed={seed}] Round {r}: starting validation eval...")
            val_losses = [eval_loss(m, val_gen, get_window_size_blocks(r, total_rounds), args.val_tokens, args.val_seq_len) for m in model_ls]
            val_ppls = [math.exp(vl) for vl in val_losses]
            avg_val = statistics.mean(val_losses)
            avg_ppl = math.exp(avg_val)
            cons_err = consensus_error(model_ls)

            row = (
                [r]
                + val_losses
                + round_losses
                + val_ppls
                + [
                    round(avg_val, 6),
                    round(avg_ppl, 4),
                    round(cons_err, 6),
                    comm_rounds_count,
                    round(cumul_time, 4),
                    round(t_elapsed, 6),
                ]
            )
            loss_table.append(row)

            if csv_path is not None:
                with open(csv_path, "w", newline="") as _cf:
                    csv.writer(_cf).writerows(loss_table)

            jwp(
                f"[seed={seed}] Round {r}/{total_rounds}: "
                f"train_loss={[round(l, 4) for l in round_losses]}, "
                f"avg_val={avg_val:.4f}, ppl={avg_ppl:.2f}, "
                f"cons_err={cons_err:.4f}, "
                f"comm_rounds={comm_rounds_count}, "
                f"time={t_elapsed:.3f}s"
            )
            if r > 10 and "test" in str(os.environ.get("JWM_COMMIT_ID", "")):
                break

    # --- iteration time statistics ---
    time_stats = {}
    if iteration_times:
        n = len(iteration_times)
        time_stats = {
            "n_iterations": n,
            "mean_sec": statistics.mean(iteration_times),
            "stdev_sec": statistics.stdev(iteration_times) if n > 1 else 0.0,
            "min_sec": min(iteration_times),
            "max_sec": max(iteration_times),
            "median_sec": statistics.median(iteration_times),
            "total_train_sec": cumul_time,
            "total_comm_rounds": comm_rounds_count,
            "bytes_per_round": bytes_per_round,
        }
        time_stats = {
            k: round(v, 6) if isinstance(v, float) else v for k, v in time_stats.items()
        }

    final_val_losses = [eval_loss(m, val_gen, get_window_size_blocks(total_rounds, total_rounds), args.val_tokens, args.val_seq_len) for m in model_ls]
    final_avg_val = statistics.mean(final_val_losses)
    final_ppl = math.exp(final_avg_val)
    final_cons_err = consensus_error(model_ls)

    final_metrics = {
        "seed": seed,
        "algorithm": alg,
        "final_avg_val_loss": round(final_avg_val, 6),
        "final_avg_val_ppl": round(final_ppl, 4),
        "final_consensus_err": round(final_cons_err, 6),
        "total_comm_rounds": comm_rounds_count,
        "total_train_sec": round(cumul_time, 4),
        "bytes_per_round": bytes_per_round,
    }

    return loss_table, final_metrics, time_stats


if __name__ == "__main__":

    jwp("Starting training")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_seq_len", type=int, default=49152)
    parser.add_argument("--val_seq_len", type=int, default=262144)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=262144)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--train_files", type=str,
                        default="/home/jinma/project_remote_jwm/remote_data/Low-rank-Muon/fineweb10B/fineweb_train_*.bin")
    parser.add_argument("--val_files", type=str,
                        default="/home/jinma/project_remote_jwm/remote_data/Low-rank-Muon/fineweb10B/fineweb_val_*.bin")
    parser.add_argument("--val_tokens", type=int, default=10485760,
                        help="Total tokens to evaluate on (default 10M, matches train_gpt_tiny.py)")
    parser.add_argument("--train_batch_size", type=int, default=8,
                        help="Gradient accumulation steps (8 matches 8-GPU train_gpt_tiny.py)")
    parser.add_argument("--num_iterations", type=int, default=5960,
                        help="Total training iterations (matches train_gpt_tiny.py)")
    parser.add_argument("--cooldown_frac", type=float, default=0.7,
                        help="Fraction of training for LR cooldown (matches train_gpt_tiny.py)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="If set, overrides num_iterations with epochs * rounds_per_epoch")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--n_workers", type=int, default=1)

    # Muon-style LR for hidden_matrix params (matches train_gpt_tiny.py Muon optimizer)
    parser.add_argument("--muon_lr", type=float, default=0.025,
                        help="Base LR for hidden matrix params (Muon-style, default 0.025)")
    parser.add_argument("--muon_wd", type=float, default=0.01,
                        help="Weight decay for hidden matrix params (default 0.01)")
    parser.add_argument("--mom", type=float, default=0.95,
                        help="Final momentum for hidden params (warmed up from 0.85 over 300 steps)")

    # Legacy LR arg for non-demuon algorithms
    parser.add_argument("--lr", type=float, default=1e-1,
                        help="Learning rate for non-demuon algorithms (dsgd, gt_dsgd, etc.)")

    parser.add_argument(
        "--msgn",
        type=int,
        default=1,
        help="0=raw gradient, 1=Newton-Schulz msgn, 2=exact SVD msgn",
    )
    parser.add_argument(
        "--ns_steps",
        type=int,
        default=5,
        help="Newton-Schulz iterations (only for --msgn=1)",
    )

    parser.add_argument(
        "--gossip_rounds",
        type=int,
        default=1,
        help="Number of gossip (mixing) rounds per iteration",
    )

    parser.add_argument(
        "--network", type=str, default="ring", choices=["ring", "exp", "complete"]
    )
    parser.add_argument("--alg", type=str, default="demuon", choices=SUPPORTED_ALGS)

    # baseline-specific args
    parser.add_argument(
        "--l2_clip_bd",
        type=float,
        default=0.1,
        help="Clipping bound for dsgd_gclip_decay",
    )
    parser.add_argument("--phi", type=float, default=1.0, help="phi parameter for sen")
    parser.add_argument("--tau", type=float, default=1.0, help="tau parameter for sen")

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="List of random seeds to run (e.g. --seeds 42 123 456)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="CUDA device index (e.g. --gpu 0). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Output directory for CSV/JSON files (created if needed)",
    )

    args = parser.parse_args()
    JWM_COMMIT_ID = str(os.environ.get("JWM_COMMIT_ID", "local"))

    # --- set device from --gpu flag ---
    if args.gpu is not None:
        _utils_mod.device = torch.device(f"cuda:{args.gpu}")
    device = _utils_mod.device
    jwp(f"Using device: {device}")

    os.makedirs(args.outdir, exist_ok=True)
    quick2json(os.path.join(args.outdir, "args.json"), vars(args))
    jwp(args)

    all_final_metrics = []

    for seed in args.seeds:
        jwp(f"\n{'='*60}")
        jwp(f"Running seed={seed}")
        jwp(f"{'='*60}")

        suffix = f"_seed{seed}" if len(args.seeds) > 1 else ""
        out_csv = Path(args.outdir) / f"loss{suffix}.csv"

        loss_table, final_metrics, time_stats = run_single_seed(
            args, seed, csv_path=str(out_csv)
        )

        all_final_metrics.append(final_metrics)
        jwp(f"[seed={seed}] Loss CSV saved to {out_csv}")

        if time_stats:
            jwp(
                f"[seed={seed}] Iteration time stats: "
                + json.dumps(time_stats, indent=2)
            )
            quick2json(str(Path(args.outdir) / f"time_stats{suffix}.json"), time_stats)

        quick2json(
            str(Path(args.outdir) / f"final_metrics{suffix}.json"), final_metrics
        )
        jwp(f"[seed={seed}] Final metrics: {json.dumps(final_metrics, indent=2)}")

    # --- multi-seed summary ---
    if len(args.seeds) > 1:
        val_losses = [m["final_avg_val_loss"] for m in all_final_metrics]
        val_ppls = [m["final_avg_val_ppl"] for m in all_final_metrics]
        cons_errs = [m["final_consensus_err"] for m in all_final_metrics]

        summary = {
            "algorithm": args.alg,
            "network": args.network,
            "n_workers": args.n_workers,
            "lr": args.lr,
            "mom": args.mom,
            "num_iterations": args.num_iterations,
            "seeds": args.seeds,
            "n_seeds": len(args.seeds),
            "val_loss_mean": round(statistics.mean(val_losses), 6),
            "val_loss_std": (
                round(statistics.stdev(val_losses), 6) if len(val_losses) > 1 else 0.0
            ),
            "val_ppl_mean": round(statistics.mean(val_ppls), 4),
            "val_ppl_std": (
                round(statistics.stdev(val_ppls), 4) if len(val_ppls) > 1 else 0.0
            ),
            "cons_err_mean": round(statistics.mean(cons_errs), 6),
            "cons_err_std": (
                round(statistics.stdev(cons_errs), 6) if len(cons_errs) > 1 else 0.0
            ),
            "per_seed_metrics": all_final_metrics,
        }
        quick2json(str(Path(args.outdir) / "multi_seed_summary.json"), summary)
        jwp(f"\n{'='*60}")
        jwp("MULTI-SEED SUMMARY")
        jwp(
            f"  Val loss: {summary['val_loss_mean']:.4f} ± {summary['val_loss_std']:.4f}"
        )
        jwp(f"  Val PPL:  {summary['val_ppl_mean']:.2f} ± {summary['val_ppl_std']:.2f}")
        jwp(
            f"  Cons err: {summary['cons_err_mean']:.4f} ± {summary['cons_err_std']:.4f}"
        )
        jwp(f"{'='*60}")
    else:
        jwp("\nSingle seed run complete. Use --seeds 42 123 456 for multi-seed runs.")
