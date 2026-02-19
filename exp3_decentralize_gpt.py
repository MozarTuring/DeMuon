from __future__ import annotations
import json
import random


from torch.utils.data import DataLoader, TensorDataset

from gpt_utils import *

from utils import *


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
        loss   = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        n      = y.numel()
        tot   += loss.item()*n
        ntok  += n
    return tot/ntok




if __name__ == '__main__':

    jwp("Starting training")
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size",type=int, default=64)
    parser.add_argument("--d_model",   type=int, default=256)
    parser.add_argument("--n_layer",   type=int, default=2)
    parser.add_argument("--n_head",    type=int, default=4)
    parser.add_argument("--max_len",   type=int, default=128)
    parser.add_argument("--train_batch_size",   type=int, default=64)
    parser.add_argument("--eval_batch_size",   type=int, default=512)
    parser.add_argument("--epochs",   type=int, default=12)
    parser.add_argument("--log_interval",   type=int, default=100)
    parser.add_argument("--n_workers",   type=int, default=8)

    parser.add_argument("--lr",   type=float, default=1e-1)
    parser.add_argument("--mom",   type=float, default=0.8)

    parser.add_argument("--msgn",   type=int, default=2)

    parser.add_argument("--lr_schedule",   type=int, default=3)
    parser.add_argument("--network",   type=str, default='complete')
    parser.add_argument("--alg",   type=str, default='demuon')

    args = parser.parse_args()
    jwm_commit_id=str(os.environ['jwm_commit_id'])

    quick2json(os.path.join('./args.json'), vars(args))
    jwp(args)

    header = ["round"]
    header += [f"w{i}_val"   for i in range(args.n_workers)]
    header += [f"w{i}_train" for i in range(args.n_workers)]
    loss_table = [header]


    filename = os.path.basename(__file__)

    wandb.init(project='neurips_code', config=args, name=f'{jwm_commit_id}')
    artifact = wandb.Artifact("my_model", type="model")
    
    loader_ls, val_loader, vocab_size, rounds_per_epoch, vocab = get_loaders(args)
    jwp(rounds_per_epoch)
    iter_ls = [iter(loader) for loader in loader_ls]
    max_round_per_epoch = max(rounds_per_epoch)
    total_rounds = args.epochs * max_round_per_epoch

    lr, mom = args.lr, args.mom
    y_list, m_list = [], []
    mixing, _ = get_graph(args, device)

    model_ls = list()
    for i in range(args.n_workers):
        model = MiniGPT(vocab_size, args.d_model, args.n_layer, args.n_head, args.max_len)
        if len(model_ls) > 0:
            model.load_state_dict(model_ls[0].state_dict())
        model.to(device)
        model_ls.append(model)
        y_list.append({name: torch.zeros_like(param) for name, param in model.named_parameters()})
        m_list.append({name: torch.zeros_like(param) for name, param in model.named_parameters()})
    aa=[(ele[0], ele[1].shape) for ele in model.named_parameters()]
    jwp(aa)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    for r in range(1, total_rounds+1):
        round_losses = []
        for wid, model in enumerate(model_ls):
            try:
                batch_x, batch_y = next(iter_ls[wid])
            except StopIteration:
                iter_ls[wid] = iter(loader_ls[wid])
                batch_x, batch_y = next(iter_ls[wid])

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # ----- forward/backward -----
            model.train()
            logits = model(batch_x)
            # jwp((logits.shape,batch_y.shape))
            loss   = loss_fn(logits.view(-1, logits.size(-1)), batch_y.view(-1))
            round_losses.append(loss.item())
            model.zero_grad(set_to_none=True)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            with torch.no_grad():
                for (name, p) in model.named_parameters():
                    if p.grad is None:
                        continue
                    m_temp = mom * m_list[wid][name]+ (1-mom) * p.grad
                    y_list[wid][name] = y_list[wid][name] + m_temp - m_list[wid][name]
                    # if name == 'pos_emb' and r > 1430:
                    #     jwp(f'{p.grad}, {p.data}')

                    m_list[wid][name] = m_temp

        
        if args.n_workers > 1:
            y_list = mix_y_list(y_list, mixing)
        if args.lr_schedule == 1:
            tmp_lr = lr * (1-r/total_rounds)
        elif args.lr_schedule == 2:
            tmp_lr = lr / math.sqrt(r)
        elif args.lr_schedule == 3:
            tmp_lr = lr
        else:
            pass
        for wid, model in enumerate(model_ls):
            with torch.no_grad():
                for (name, p) in model.named_parameters():
                    if p.grad is None:
                        continue
                    tmp_shape = y_list[wid][name].shape
                    tmp = y_list[wid][name].squeeze()
                    if tmp.ndim == 0:
                        denominator = torch.abs(tmp)
                        assert denominator != 0
                        p.data -= tmp_lr  * tmp.reshape(tmp_shape)  / denominator
                    elif tmp.ndim == 1:
                        denominator = torch.norm(tmp)
                        assert denominator != 0
                        p.data -= tmp_lr  * tmp.reshape(tmp_shape)  / denominator
                    elif tmp.ndim == 2:
                        if args.msgn == 0:
                            p.data -= tmp_lr  * y_list[wid][name]
                        elif args.msgn == 1:
                            update = zeropower_via_newtonschulz5(tmp)
                            p.data -= tmp_lr  * update.reshape(tmp_shape)
                            
                        elif args.msgn == 2:
                            U, S, Vt = torch.linalg.svd(tmp, full_matrices=False)
                            update = U @ Vt
                            p.data -= tmp_lr  * update
                    else:
                        jwp(f'{name}, error, {tmp}')
                        1/0

                    check_nan_inf(name, tmp, r)
        if args.n_workers > 1:
            mix_params(model_ls, mixing)
    
        if r % args.log_interval == 0 or r == total_rounds or r == 1:
            val_losses = [eval_loss(m, val_loader, loss_fn) for m in model_ls]
            loss_table.append([r] + val_losses + round_losses)
            wandb.log({'training loss': np.average(round_losses)}, step=r)
            jwp(f"Round {r}/{total_rounds}: {round_losses}")
            # wandb.log({'lr': lr/r, 'lr_norm': lr/r/denominator}, step=r)
            if r > 10 and "test" in jwm_commit_id:
                break

    out_csv = Path(f"./loss.csv")
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(loss_table)
