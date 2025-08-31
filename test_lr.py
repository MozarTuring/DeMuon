# gaussian noise, ring graph
from lr_gt_token import *
from graph import connected_cycle_weights
import os

from tools_jw import *

os.makedirs("graphs", exist_ok=True)
beta_ls=[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]# 11 in total

weights = connected_cycle_weights(filename=f"graphs/ring_20.npy", n=20, degree=1)
print(weights)
alpha_ls=[1e-2]#10,5,1,0.5,1e-1,5e-2,1e-2,5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5  # alpha is learning rate 13 in total
for alpha in alpha_ls:
    plt.figure()
    for beta in beta_ls:
        run_datasets_alg_network(dataset="token", noise="gaussian", noise_scale=3, alg="gt_nsgdm", paras=[weights,alpha,beta],
                             num_rep=1, num_steps=10000, seed=24, num_clients=20)
        print("Finished beta=",beta)
    plt.legend()
    plt.grid(True)
    plt.yticks(np.arange(0, 1, 0.1))
    plt.ylim(0,1)
    os.makedirs("figs", exist_ok=True)
    if os.path.exists(f'figs/alpha{alpha}.pdf'):
        os.remove(f'figs/alpha{alpha}.pdf')
    plt.savefig(f'figs/alpha{alpha}.pdf')
    plt.close()

print("\a")

# !pip install -r requirements_lr.txt
