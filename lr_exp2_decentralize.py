from __future__ import annotations
import random


from torch.utils.data import DataLoader, TensorDataset


from utils import *






if __name__ == '__main__':

    jwp("Starting training")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int, required=True)
    parser.add_argument("--train_batch_size",type=int, required=True)
    parser.add_argument("--eval_batch_size",type=int, required=True)
    parser.add_argument("--log_interval",type=int, required=True)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--network", type=str, required=True)

    args = parser.parse_args(['--train_batch_size','128','--eval_batch_size','3500', '--epochs','2000', '--log_interval', '100', '--n_workers', '8', '--network', 'complete'])

    filename = os.path.basename(__file__)

    wandb.init(project='neurips_code', name=f'{filename}')
    artifact = wandb.Artifact("my_model", type="model")
    

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)


    g = torch.Generator()
    g.manual_seed(args.random_seed)

    x=torch.tensor([[2.5],[1.5]])
    loader_ls = list()
    iter_ls = list()
    for i in range(args.n_workers):
        data_x=torch.randn([1000,2])
        data_y=data_x@x
        dataset = TensorDataset(data_x, data_y)
        loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, generator=g)
        loader_ls.append(loader)
        iter_ls.append(iter(loader))

    max_round_per_epoch = len(loader)
    total_rounds = args.epochs * max_round_per_epoch


    args.lr=1e-1
    args.mom=0.8
    lr, mom = args.lr, args.mom
    y_list, m_list = [], []
    mixing, _ = get_graph(args, device)

    model_ls = list()
    for i in range(args.n_workers):
        model = nn.Linear(2,1,bias=False)
        if len(model_ls) > 0:
            model.load_state_dict(model_ls[0].state_dict())
        model.to(device)
        model_ls.append(model)
        y_list.append({name: torch.zeros_like(param) for name, param in model.named_parameters()})
        m_list.append({name: torch.zeros_like(param) for name, param in model.named_parameters()})
    aa=[(ele[0], ele[1].shape) for ele in model.named_parameters()]
    jwp(aa)
    loss_fn = nn.MSELoss()

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
            loss   = loss_fn(logits, batch_y)
            round_losses.append(loss.item())
            model.zero_grad(set_to_none=True)
            loss.backward()

            with torch.no_grad():
                for (name, p) in model.named_parameters():
                    if p.grad is None:
                        continue
                    m_temp = mom * m_list[wid][name]+ (1-mom) * p.grad
                    y_list[wid][name] = y_list[wid][name] + m_temp - m_list[wid][name]
                    m_list[wid][name] = m_temp

        
        
        y_list = mix_y_list(y_list, mixing)
        for wid, model in enumerate(model_ls):
            with torch.no_grad():
                for (name, p) in model.named_parameters():
                    if p.grad is None:
                        continue
                    tmp_shape = y_list[wid][name].shape
                    tmp = y_list[wid][name].squeeze()
                    if tmp.ndim == 0:
                        denominator = torch.abs(tmp)
                    elif tmp.ndim == 1:
                        denominator = torch.norm(tmp)
                    else:
                        jwp(f'{name}, error, {tmp}')
                        1/0
                    assert denominator != 0
                    p.data -= lr * (1-r/total_rounds)  * tmp.reshape(tmp_shape)  / denominator

        mix_params(model_ls, mixing)
    

        if r % args.log_interval == 0 or r == total_rounds or r == 1:
            wandb.log({'training loss': np.average(round_losses)}, step=r)
            jwp(f"Round {r}/{total_rounds}: {round_losses}")
            # wandb.log({'lr': lr/r, 'lr_norm': lr/r/denominator}, step=r)
