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

    args = parser.parse_args(['--train_batch_size','128','--eval_batch_size','3500', '--epochs','2000', '--log_interval', '100'])

    wandb.init(project='neurips_code', name='lr_exp')
    artifact = wandb.Artifact("my_model", type="model")
    

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)


    g = torch.Generator()
    g.manual_seed(args.random_seed)

    x=torch.tensor([[2.5],[1.5]])
    data_x=torch.randn([1000,2])
    data_y=data_x@x
    dataset = TensorDataset(data_x, data_y)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, generator=g)

    max_round_per_epoch = len(loader)
    total_rounds = args.epochs * max_round_per_epoch


    args.lr=1e-1
    args.mom=0.8
    lr, mom = args.lr, args.mom

    model = nn.Linear(2,1,bias=False)
    aa=[(ele[0], ele[1].shape) for ele in model.named_parameters()]
    jwp(aa)
    loss_fn = nn.MSELoss()

    model.to(device)

    y_list = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    m_list = {name: torch.zeros_like(param) for name, param in model.named_parameters()}


    tmp_iter = iter(loader)
    for r in range(1, total_rounds+1):
        round_losses = []
        try:
            batch_x, batch_y = next(tmp_iter)
        except StopIteration:          # start a new local epoch
            tmp_iter = iter(loader) # reset the loader
            batch_x, batch_y = next(tmp_iter)

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
                m_temp = mom * m_list[name]+ (1-mom) * p.grad
                y_list[name] = y_list[name] + m_temp - m_list[name]
                m_list[name] = m_temp

                if y_list[name].ndim == 0:
                    denominator = torch.abs(y_list[name])
                else:
                    denominator = torch.norm(y_list[name])
                assert denominator != 0

                # denominator = 1

                # p.data -= lr / math.sqrt(r) * y_list[name]  / denominator
                p.data -= lr * (1-r/total_rounds)  * y_list[name]  / denominator
            

        if r % args.log_interval == 0 or r == total_rounds or r == 1:
            wandb.log({'training loss': round_losses[0]}, step=r)
            # wandb.log({'lr': lr/r, 'lr_norm': lr/r/denominator}, step=r)
