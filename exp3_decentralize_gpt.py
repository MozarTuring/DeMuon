from __future__ import annotations
import json
import random
import statistics
import time
import math

from torch.utils.data import DataLoader, TensorDataset

import utils as _utils_mod
import gpt_utils as _gpt_utils_mod
from gpt_utils import *
from utils import *

SUPPORTED_ALGS = ["demuon", "dsgd", "dsgd_gclip_decay", "gt_dsgd", "gt_nsgdm", "sen"]


def quick2json(inp_path, inp_data):
    with open(inp_path, "w", encoding="utf8") as wf:
        wf.write(json.dumps(inp_data, ensure_ascii=False, indent=2))


@torch.no_grad()
def eval_loss(model, loader, loss_fn):
    model.eval()
    tot, ntok = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        n = y.numel()
        tot += loss.item() * n
        ntok += n
    return tot / ntok


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

    loader_ls, val_loader, vocab_size, rounds_per_epoch, vocab = get_loaders(args)
    jwp(f"[seed={seed}] rounds_per_epoch={rounds_per_epoch}")

    # --- measure data heterogeneity (once per seed) ---
    tok_tokenizer = get_tokenizer("basic_english")
    from torchtext.datasets import Multi30k

    partitions_tokens = []
    all_tokens = []
    for eng, _de in Multi30k(split="train", language_pair=("en", "de")):
        all_tokens.extend(vocab(tok_tokenizer(eng.lower())))
    random.shuffle(all_tokens)
    part_len = len(all_tokens) // args.n_workers
    for i in range(args.n_workers):
        partitions_tokens.append(all_tokens[i * part_len : (i + 1) * part_len])
    het_kl = measure_data_heterogeneity(partitions_tokens, vocab_size)
    jwp(f"[seed={seed}] Data heterogeneity (mean sym-KL): {het_kl:.6f}")

    iter_ls = [iter(loader) for loader in loader_ls]
    max_round_per_epoch = max(rounds_per_epoch)
    total_rounds = args.epochs * max_round_per_epoch

    lr, mom = args.lr, args.mom
    mixing, _ = get_graph(args, device)

    model_ls = list()
    for i in range(args.n_workers):
        model = MiniGPT(
            vocab_size, args.d_model, args.n_layer, args.n_head, args.max_len
        )
        if len(model_ls) > 0:
            model.load_state_dict(model_ls[0].state_dict())
        model.to(device)
        model_ls.append(model)

    ref_model = model_ls[-1]
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    # --- algorithm-specific buffer init ---
    alg = args.alg
    y_list, m_list, g_prev_list = [], [], []

    if alg in ("demuon", "gt_nsgdm"):
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
    round0_train_losses = []
    for wid, model in enumerate(model_ls):
        batch_x, batch_y = next(iter_ls[wid])
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        model.eval()
        with torch.no_grad():
            logits = model(batch_x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), batch_y.view(-1))
        round0_train_losses.append(loss.item())
    # reset iterators so round 1 sees the same batches
    iter_ls = [iter(loader) for loader in loader_ls]

    val_losses_0 = [eval_loss(m, val_loader, loss_fn) for m in model_ls]
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

    for r in range(1, total_rounds + 1):
        t_start = time.perf_counter()
        round_losses = []

        # ===== per-worker forward/backward + local buffer update =====
        for wid, model in enumerate(model_ls):
            try:
                batch_x, batch_y = next(iter_ls[wid])
            except StopIteration:
                iter_ls[wid] = iter(loader_ls[wid])
                batch_x, batch_y = next(iter_ls[wid])

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            model.train()
            logits = model(batch_x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), batch_y.view(-1))
            round_losses.append(loss.item())
            model.zero_grad(set_to_none=True)
            loss.backward()

            with torch.no_grad():
                if alg == "demuon":
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        m_temp = mom * m_list[wid][name] + (1 - mom) * p.grad
                        y_list[wid][name] = (
                            y_list[wid][name] + m_temp - m_list[wid][name]
                        )
                        m_list[wid][name] = m_temp

                elif alg == "dsgd":
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        p.data -= tmp_lr * p.grad

                elif alg == "dsgd_gclip_decay":
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
                        m_temp = m_list[wid][name].mul(mom).add(g, alpha=1 - mom)
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
                        m.mul_(mom / r**0.5).add_(temp, alpha=1 - mom / r**0.5)
                        m_list[wid][name] = m
                        p.data -= lr / r**0.2 * m

        # ===== stepsize schedule =====
        if args.lr_schedule == 1:
            tmp_lr = lr * (1 - r / total_rounds)
        elif args.lr_schedule == 2:
            tmp_lr = lr / math.sqrt(r)
        elif args.lr_schedule == 3:
            tmp_lr = lr
        elif args.lr_schedule == 4:
            tmp_lr = lr / r
        elif args.lr_schedule == 5:
            tmp_lr = lr / (r ** 0.75)
        else:
            exit("error")

        # ===== mixing / communication =====
        if alg == "demuon":
            if args.n_workers > 1:
                for _ in range(args.gossip_rounds):
                    y_list = mix_y_list(y_list, mixing)
                    comm_rounds_count += 1

            for wid, model in enumerate(model_ls):
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        tmp_shape = y_list[wid][name].shape
                        tmp = y_list[wid][name].squeeze()
                        if tmp.ndim == 0:
                            p.data -= tmp_lr * tmp.reshape(tmp_shape) / torch.abs(tmp)
                        elif tmp.ndim == 1:
                            p.data -= tmp_lr * tmp.reshape(tmp_shape) / torch.norm(tmp)
                        elif tmp.ndim == 2:
                            if args.msgn == 0:
                                p.data -= tmp_lr * y_list[wid][name]
                            elif args.msgn == 1:
                                update = zeropower_via_newtonschulz5(
                                    tmp, steps=args.ns_steps
                                )
                                p.data -= tmp_lr * update.reshape(tmp_shape)
                            elif args.msgn == 2:
                                U, S, Vt = torch.linalg.svd(tmp, full_matrices=False)
                                p.data -= tmp_lr * (U @ Vt)
                        else:
                            jwp(f"{name}, error, {tmp}")
                            1 / 0
                        check_nan_inf(name, tmp, r)

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
            for wid, model in enumerate(model_ls):
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        p.data -= lr * y_list[wid][name]
            if args.n_workers > 1:
                mix_params(model_ls, mixing)
                comm_rounds_count += 1

        elif alg == "gt_nsgdm":
            if args.n_workers > 1:
                y_list = mix_y_list(y_list, mixing)
                comm_rounds_count += 1
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

        if r % args.log_interval == 0 or r == total_rounds or r == 1:
            val_losses = [eval_loss(m, val_loader, loss_fn) for m in model_ls]
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

    final_val_losses = [eval_loss(m, val_loader, loss_fn) for m in model_ls]
    final_avg_val = statistics.mean(final_val_losses)
    final_ppl = math.exp(final_avg_val)
    final_cons_err = consensus_error(model_ls)

    final_metrics = {
        "seed": seed,
        "algorithm": alg,
        "final_avg_val_loss": round(final_avg_val, 6),
        "final_avg_val_ppl": round(final_ppl, 4),
        "final_consensus_err": round(final_cons_err, 6),
        "data_heterogeneity_kl": round(het_kl, 6),
        "total_comm_rounds": comm_rounds_count,
        "total_train_sec": round(cumul_time, 4),
        "bytes_per_round": bytes_per_round,
    }

    return loss_table, final_metrics, time_stats


if __name__ == "__main__":

    jwp("Starting training")
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--n_workers", type=int, default=8)

    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--mom", type=float, default=0.8)

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
        "--lr_schedule",
        type=int,
        default=3,
        help="1=linear decay, 2=1/sqrt(t), 3=constant, 4=1/t, 5=1/t^0.75",
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
            "epochs": args.epochs,
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
