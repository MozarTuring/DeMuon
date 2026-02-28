# How Other Decentralized Training Methods Perform on GPT-2–Style Models (Open Literature)

Summary of how various decentralized / federated / low-communication training methods perform on transformer language models (GPT-2–scale and similar), based on published work.

---

## 1. DiLoCo (Distributed Low-Communication) — Federated-Averaging Style

**Reference:** *DiLoCo: Distributed Low-Communication Training of Language Models* (DeepMind, arXiv:2311.08105).

- **Setup:** Decoder-only transformers (60M, 150M, 400M params, Chinchilla-style), C4 dataset, **8 workers**, non-IID data (clustered by pretrained features).
- **Mechanism:** FedAvg-style: each worker runs **AdamW** for **H inner steps** (e.g. 500), then **outer Nesterov momentum** on averaged parameter deltas; sync every H steps.
- **Performance (150M, 8 workers, 64k steps after 24k pretrain):**
  - **Perplexity:** DiLoCo **15.02** vs single-worker baseline **16.23** (better), vs 8× batch data-parallel **15.30** (DiLoCo better), vs 8× updates **14.72** (more compute).
  - **Communication:** **~500× less** than per-step gradient sync; same wall-clock as 1× baseline, 8× compute/data.
- **Robustness:** Works from scratch or with pretraining; robust to non-IID; H=500–1000 gives good trade-off; tolerant to **dropped sync** (e.g. 50% drop ≈ 2.1% PPL degradation); **adaptive compute** (varying number of workers over time) works; 4–64 replicas tested.
- **Scaling:** 60M / 150M / 400M all improve with DiLoCo (e.g. 4–7.5% relative PPL gain over single worker). Later scaling-law work (e.g. arXiv:2503.09799) suggests DiLoCo can scale better than data-parallel as model size grows.

**Takeaway:** Strong reference for “decentralized” (multi-island) training of GPT-2–scale LMs: same or better perplexity than sync training with 500× less communication.

---

## 2. High-Performance Decentralized Training (ICLR’25) — Peer-to-Peer, GPT-2 on OpenWebText

**Reference:** *From Promise to Practice: Realizing High-Performance Decentralized Training* (Wang et al., ICLR’25; GitHub: WangZesen/Decentralized-Training-Exp).

- **Setup:** **GPT-2 on OpenWebText**; also ResNet-50 (ImageNet), Transformer NMT (WMT14).
- **Methods:** Decentralized optimizers **without** a central server, including **AccumAdam** (overlap compute/communication, good generalization).
- **Result:** Decentralized training reaches **generalization comparable to All-Reduce** while improving scalability and reducing communication; code in `gpt2/` in the repo.

**Takeaway:** Direct evidence that peer-to-peer decentralized methods can match centralized (All-Reduce) training on GPT-2 pretraining (OpenWebText).

---

## 3. FedAvg / Local SGD on Language Models

- **Classic FedAvg (McMahan et al.):** Parameter averaging every H steps; outer optimizer = SGD (no momentum). DiLoCo paper reports **FedAvg (outer SGD) and FedOpt (outer Adam) performed poorly** for their 150M transformer (instability for Adam); **Nesterov outer** worked best.
- **FedMom (Nesterov outer):** Huo et al. use Nesterov as outer optimizer; evaluated on **small LSTM LM**, 2 replicas, communication every ~20 steps. DiLoCo scales this to 150M–400M transformers, 8–64 replicas, H=500–1000.
- **Cross-device FL, 21M transformer:** Prior work (e.g. FedNLP-style) shows **~11% perplexity gain** over smaller LSTM with **~10× less client–server communication**; 21M transformer comparable or better perplexity than smaller baselines.

**Takeaway:** FedAvg/local-SGD can work for LMs, but outer optimizer (e.g. Nesterov) and large H are important at scale; DiLoCo is the current benchmark for FedAvg-style methods on GPT-2–scale.

---

## 4. D-PSGD / Ring / Exponential Graph (Generic Decentralized SGD)

- **D-PSGD / D²:** Standard decentralized SGD (and D² for heterogeneous data) is well studied for **CV (e.g. ResNet, image classification)** and theoretically; **ring** and **exponential graphs** are common. Exponential graphs give **Õ(1) per-iteration communication** and good convergence; one-peer exponential can match static exponential in convergence while being more communication-efficient.
- **Transformers/GPT-2:** The ICLR’25 decentralized training work (above) shows that **decentralized (peer-to-peer) methods** can be applied to **GPT-2 on OpenWebText** with parity to All-Reduce; topology and optimizer (e.g. AccumAdam) matter for performance.

**Takeaway:** Ring/exponential and similar topologies are used in decentralized deep learning; on GPT-2, recent benchmarks (e.g. ICLR’25) show they can match centralized training when combined with suitable optimizers.

---

## 5. Communication-Efficient and Asynchronous Variants

- **A²CiD²:** Asynchronous randomized gossip with continuous local momentum; tested on **64 A100 GPUs**; reduces communication and improves efficiency in poorly connected networks.
- **GWTF (Go With The Flow):** Churn-tolerant decentralized training for **LLMs**; reduces training time by **up to ~45%** in heterogeneous, high-churn settings.
- **Consensus control:** Keeping consensus distance below a threshold yields convergence similar to centralized; relevant for any decentralized method (including when applied to transformers).

**Takeaway:** Asynchronous and churn-tolerant decentralized methods are being applied to LLM-scale training; concrete GPT-2 numbers are less standardized than DiLoCo and ICLR’25.

---

## 6. FSDP / DDP (Centralized Data Parallel — for comparison)

**Reference:** *A Comparative Analysis of Distributed Training Strategies for GPT-2* (arXiv:2405.15628).

- **Scope:** **Data and model parallelism** (FSDP, DDP), not peer-to-peer decentralized. Compares single-GPU vs FSDP vs DDP for GPT-2 in terms of training time, memory, throughput, loss, grad norm.
- **Takeaway:** For **centralized** multi-GPU training of GPT-2, FSDP and DDP are the standard baselines; your setting (DeMuon, ring/exp graph) is **decentralized** (no central server), so DiLoCo and ICLR’25 are closer comparisons.

---

## Summary Table (GPT-2–scale LMs)

| Method              | Setting           | Communication      | Main result (PPL / quality)                    |
|---------------------|-------------------|--------------------|-----------------------------------------------|
| **DiLoCo**          | 8 workers, C4     | ~500× less         | 15.02 PPL (better than 1× and 8× batch sync)  |
| **ICLR’25 decentralized** | GPT-2, OpenWebText | Reduced vs All-Reduce | Comparable to All-Reduce                      |
| **FedAvg (outer SGD)** | 150M, C4        | Periodic            | Poor in DiLoCo experiments                    |
| **FedOpt (outer Adam)** | 150M, C4       | Periodic            | Unstable unless heavily regularized           |
| **FedMom (Nesterov)**  | Small LSTM      | Every ~20 steps     | Good; DiLoCo extends to large transformers   |

---

## Practical recommendations for your DeMuon + GPT-2 setup

1. **Compare against DiLoCo** if you have a multi-island / low-communication regime: same “periodic sync” idea, but you use **peer-to-peer mixing (ring/exp)** and **DeMuon (momentum + signed updates)** instead of a central Nesterov outer step.
2. **Reproduce ICLR’25 GPT-2 experiments** (Decentralized-Training-Exp, `gpt2/`) for a direct peer-to-peer baseline on OpenWebText.
3. **Topology:** Exponential graph is provably efficient for decentralized deep training; your `--network exp` is aligned with that.
4. **Outer step / sync frequency:** DiLoCo uses H=500–1000; you can compare different round lengths (e.g. rounds per communication) to trade off communication vs convergence.
5. **Data partition:** DiLoCo shows robustness to non-IID; your Multi30k partition across workers is a reasonable stand-in for heterogeneous data.

---

## References (key papers)

- DiLoCo: arXiv:2311.08105 (v3).
- DiLoCo scaling: arXiv:2503.09799.
- ICLR’25 decentralized (GPT-2, OpenWebText): arXiv:2410.11998; GitHub: WangZesen/Decentralized-Training-Exp.
- D² (decentralized data): arXiv:1803.07068.
- Exponential graph decentralized: NeurIPS 2021, “Exponential Graph is Provably Efficient for Decentralized Deep Training”.
- FSDP/DDP vs GPT-2: arXiv:2405.15628.
- FedAvg: McMahan et al., “Communication-Efficient Learning of Deep Networks from Decentralized Data”.
- FedMom (Nesterov): Huo et al., FedMom.
