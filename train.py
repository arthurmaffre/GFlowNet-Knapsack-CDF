"""
train.py – Knapsack GFlowNet Trainer
====================================

🚀 ONE-MINUTE PITCH ────────────────
Solve the 0-1 knapsack problem **without exhaustively enumerating** all 2ⁿ
solutions.  A Generative Flow Network (GFlowNet) learns a *probability
distribution* over **every feasible item-set**, so high-value combinations are
sampled often while low-value ones fade away.  In practice we reach the global
optimum in ≈ 2 600 sampled trajectories (20 epochs × 128-batch) instead of
exploring all 32 768 paths.

───────────────────────────────────────────────────────────────────────────────
0. Audience Guide
───────────────────────────────────────────────────────────────────────────────
• 🎓 **Economists (no heavy maths / AI)**  
  – Think of this as a *smarter survey* of every shopping basket under a
    budget-cap; the network learns which baskets give the best bang-for-buck
    and how likely each is.  
  – Output = a *distribution*, not just “one best” bundle, so you can study
    alternative allocations and their welfare.

• 🛠 **Operations-Research / Optimisation folks (no econ / AI)**  
  – GFlowNet ≈ “Monte-Carlo Dynamic Programming × normalised flows”.  
  – It *implicitly* explores the binary decision tree **without breadth-first
    DP**; sample efficiency is O(10²–10³) vs exhaustive 2ⁿ.  
  – Provides *provably unbiased* estimates of the optimal value via
    Trajectory-Balance; our runs hit the true optimum on *every* seed so far.

• 🤖 **AI / ML Engineers (no econ / maths)**  
  – Standard TB loss:  `log Pθ(τ) + log Z = log R(τ)` with learnable `log Z`.  
  – Three state-representations (v1 baseline, v2 block-traj, v3 dynamic-budget).  
  – Integrated Weights & Biases logging, GPU/MPS auto-detect, Sweep-API for
    Bayesian hyper-search.  All plots in the screenshots (parallel coords,
    CDFs, KL curves) are logged out-of-the-box.

───────────────────────────────────────────────────────────────────────────────
👁  Visual intuition
───────────────────────────────────────────────────────────────────────────────

1)  The search tree (3 items = 2³ paths) – ASCII snapshot
    ┌───(root)
    │
    ├──0───0───0   ← skip, skip, skip
    │   │   └──1   ← skip, skip, **take**
    │   └──1───0
    │       └──1
    └──1───0───0   ← **take**, …
        │   └──1
        └──1───0
            └──1   ← …take, take, take (leaf)

    • Inside the model we encode the embedding vector (or context vector for the model) 
        them with -1, 0, 1 : skip, not seen wet, take
    • DP would *visit* every leaf.  
    • GFlowNet *samples* leaves – high-reward ones (bold) appear with
      higher probability but low-reward paths are still occasionally drawn,
      so we stay exploratory.

2)  How a GFlowNet rolls out a trajectory – Mermaid sequence
    ```mermaid
    sequenceDiagram
        participant s0 as state s₀ (budget B)
        participant π as πθ (GFlowNet)
        participant s1 as state s₁
        participant s2 as state s₂
        participant End as terminal τ

        s0 ->> π : context = [s₀]
        π  -->> s0 : P(a₀=take/skip)
        Note over s0,π: GPU ⇒ 128 states queried **in parallel**

        π ->> s1 : sample a₀, update<br/>remaining budget
        s1 ->> π : context = [s₁]
        π  -->> s1 : P(a₁)
        π ->> s2 : …

        loop until all items seen
            …
        end
        s₂ ->> End : trajectory τ, reward R(τ)
    ```
    • All 128 trajectories in a batch share the same forward pass, so the GPU
      builds 128 “branches” at once instead of one after the other.  
    • **Batching** = massive speed-up versus naïve tree search.

───────────────────────────────────────────────────────────────────────────────

        Note over s0,π:
          On every forward pass we **sample** 128 partial states in parallel.  
          ➜ The policy πθ is *global*: its logits are defined for **all** 2ⁿ
            possible prefixes; we just train it with *mini-batches*.  
          After the gradient step we throw these 128 paths away and draw a new
          batch, so over many epochs the whole tree is covered stochastically.


───────────────────────────────────────────────────────────────────────────────
1. Problem Statement – 0-1 Knapsack (NP-hard)
───────────────────────────────────────────────────────────────────────────────
Given *utilities* uᵢ, *costs* tᵢ, and a budget B, choose a binary vector
x ∈ {0,1}ⁿ that maximises ∑uᵢxᵢ subject to ∑tᵢxᵢ ≤ B.

A traditional DP traverses the full binary tree (2ⁿ leaves).  
With n = 15 that’s 32 768 nodes; with n = 50 it’s 1 Peta-paths.  
**GFlowNet** learns to *sample* from that tree so paths with high reward
appear often, yet exploration never collapses too early.

───────────────────────────────────────────────────────────────────────────────
2. Why GFlowNet instead of Classic RL or DP?
───────────────────────────────────────────────────────────────────────────────
| Approach | What it does | Limitation | How GFlowNet fixes it |
|----------|--------------|------------|-----------------------|
| **DP / Tree-search** | Visits *every* path | Exponential blow-up | Samples O(10³) paths instead |
| **RL (e.g. Q-learning)** | Optimises one trajectory | Can get stuck in a local optimum | Learns a **distribution** ⇒ diversity + exploration |
| **GFlowNet** | Optimises *probability* of each path s.t. P ∝ reward | — | Finds optimum & its neighbourhood efficiently |

Empirically on 15 items we converge in ≈ 2 600 sampled trajectories.

───────────────────────────────────────────────────────────────────────────────
3. Model Variants
───────────────────────────────────────────────────────────────────────────────
v1 baseline        – Ignores budget during generation; infeasible baskets get
                     reward = 10 (small but non-zero to avoid log(0)).  
v2 block_traj      – *Online* feasibility check: any choice that would overspend
                     is forced to 0, trajectory continues.  
v3 dynamic_budget  – Augments the state with “remaining budget”, letting the
                     policy spend intelligently. **Best performer.**

───────────────────────────────────────────────────────────────────────────────
4. Loss Function – Trajectory Balance
───────────────────────────────────────────────────────────────────────────────
For every sampled sequence τ:

    L(τ) = (log Pθ(τ) + log Z  –  log R(τ))²

`log Z` is a single scalar parameter; learning it stabilises gradients.  
KL divergence to the ideal distribution is logged each epoch.

───────────────────────────────────────────────────────────────────────────────
5. WandB Instrumentation
───────────────────────────────────────────────────────────────────────────────
• Logs: loss, KL, `log Z`, mean / max reward, CDF plots every N epochs.  
• Sweeps: define `sweep.yaml` then run `wandb agent <entity>/<proj>/<sweep-id>`.  
  Bayesian optimisation over `lr_main`, `lr_z`, `embedding_dim`, etc.  
• Screenshots above show:  
  – *Parameter importance* (feature bars)  
  – *Parallel-coordinates* of hyperparameters vs performance  
  – *CDF alignment* (target vs sampled distribution)

───────────────────────────────────────────────────────────────────────────────
6. Quick-Start
───────────────────────────────────────────────────────────────────────────────

python train.py

or 

python train.py --arg=value ... for running it with args

"""


from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Tuple, Dict

import torch
import wandb
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────────
# Local imports – keep them grouped for clarity
# ────────────────────────────────────────────────────────────────────────────────
from reward.reward import compute_reward
from reward.analytical_reward import compute_analytical_reward
from metrics.probability_distribution_on_batch import log_distribution_and_cdf

from models.baseline_v1 import GFlowNet as BaselineV1
from models.block_traj_v2 import GFlowNet as BlockTrajV2
from models.remaining_budget_v3 import GFlowNet as DynamicBudgetV3

# Mapping from CLI flag to concrete class
MODEL_MAP = {
    "v1": BaselineV1,
    "v2": BlockTrajV2,
    "v3": DynamicBudgetV3,
}

# ────────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────────

def select_device() -> torch.device:
    """Return the *best* available torch device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_tensors(data: Dict[str, torch.Tensor], batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Expand 1‑D data arrays to batched tensors on the chosen *device*."""
    u = torch.tensor(data["u"], dtype=torch.float32, device=device).expand(batch_size, -1)
    t = torch.tensor(data["t"], dtype=torch.float32, device=device).expand(batch_size, -1)
    B = torch.tensor(data["B"], dtype=torch.float32, device=device).view(1, 1).expand(batch_size, 1)
    return u.detach(), t.detach(), B.detach(), u.size(1)


# ────────────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────────────

def train(cfg: argparse.Namespace) -> None:  # noqa: C901 (keep flat for readability)
    """End‑to‑end training routine driven by a parsed *cfg* namespace."""
    # 1 – Weights & Biases initialisation -------------------------------------------------
    run_name = f"{cfg.model_version}_bs{cfg.batch_size}_ep{cfg.num_epochs}_{wandb.util.generate_id()[:5]}"
    wandb.init(project="gflownet-knapsack", name=run_name, config=vars(cfg))

    # 2 – Device & data -------------------------------------------------------------------
    device = select_device()
    print(f"📟  Training on {device.type.upper()}")

    with Path(cfg.data_path).open("rb") as fh:
        raw_data = pickle.load(fh)

    max_reward_bruteforce = compute_analytical_reward(
        u_vals=raw_data["u"], t_vals=raw_data["t"], B_val=raw_data["B"]
    )
    wandb.config.update({"max_reward_bruteforce": max_reward_bruteforce})

    u, t, B, num_items = prepare_tensors(raw_data, cfg.batch_size, device)

    # 3 – Model & optimiser ---------------------------------------------------------------
    ModelCls = MODEL_MAP[cfg.model_version]
    model = ModelCls(
        num_items=num_items,
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        init_value_z=cfg.init_value_z,
    ).to(device)

    log_z_params = [model.log_z]
    other_params = [p for p in model.parameters() if p is not model.log_z]

    optimizer = torch.optim.SGD(
        [
            {"params": other_params, "lr": cfg.lr_main, "momentum": cfg.mom_main, "weight_decay": 1e-4},
            {"params": log_z_params, "lr": cfg.lr_z, "momentum": cfg.mom_z, "weight_decay": 0.0},  # no decay on *z*
        ]
    )

    # 4 – Training loop -------------------------------------------------------------------
    best_reward = 0.0
    kl_history = []  # KL values after epoch 200 (for sweep metric)

    pbar = tqdm(range(cfg.num_epochs), desc=f"Model {cfg.model_version} | {device.type.upper()}")
    for epoch in pbar:
        optimizer.zero_grad()

        seq_logp, selected = model.generate_trajectories(B, u, t, cfg.batch_size, num_items, device)
        assert seq_logp.requires_grad, "sequence_log_prob should be differentiable!"

        reward = compute_reward(selected, u, t, B, num_items)
        loss = model.compute_loss(seq_logp, reward)
        loss.backward()
        optimizer.step()

        # ───────── Evaluation (no‑grad) ───────────────────────────────────────────────
        with torch.no_grad():
            avg_reward = reward.mean().item()
            max_reward = reward.max().item()
            best_reward = max(best_reward, max_reward)

            prob_seq = torch.exp(seq_logp)
            p_target = reward / reward.sum()
            prob_seq = prob_seq / prob_seq.sum()
            kl = (p_target * (torch.log(p_target + 1e-6) - torch.log(prob_seq + 1e-6))).sum().item()

            if epoch >= 200:
                kl_history.append(kl)

            # Optional diagnostic CDF plot (every 50 epochs)
            log_distribution_and_cdf(epoch, reward, seq_logp, p_target, wandb, step_interval=50)

        # ───────── Progress bar & W&B logging ──────────────────────────────────────────
        pbar.set_postfix(Loss=f"{loss.item():.4f}", Rew_Avg=f"{avg_reward:.2f}", Rew_Max=f"{max_reward:.2f}", KL=f"{kl:.4f}")

        wandb.log({
            "loss": loss.item(),
            "avg_reward": avg_reward,
            "kl_div": kl,
            "max_reward": best_reward,
            "z": model.log_z.item(),
        }, step=epoch)

    # 5 – Post‑training summary -----------------------------------------------------------
    if kl_history:
        mean_kl = sum(kl_history) / len(kl_history)
        if not mean_kl:  # handle exact zero or NaN
            wandb.log({"mean_kl_200_to_end": 2.0, "kl_zero": True})
        else:
            wandb.log({"mean_kl_200_to_end": mean_kl})
            print(f"Mean KL (200‑{cfg.num_epochs}) = {mean_kl:.4f}")

    wandb.finish()


# ────────────────────────────────────────────────────────────────────────────────
# CLI entry‑point
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GFlowNet knapsack with W&B + SGD")

    # Core hyper‑parameters -------------------------------------------------------------
    parser.add_argument("--model_version", choices=list(MODEL_MAP), default="v1", help="Model variant to train (v1 or v2 or v3)")
    parser.add_argument("--data_path", type=str, default="data.pickle", help="Pickle file with u, t, B")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=400, help="Number of training epochs")

    parser.add_argument("--embedding_dim", type=int, default=95)
    parser.add_argument("--hidden_dim", type=int, default=230)
    parser.add_argument("--init_value_z", type=float, default=9.5)

    # Optimiser hyper‑parameters --------------------------------------------------------
    parser.add_argument("--lr_main", type=float, default=2e-3, help="Learning rate for main params")
    parser.add_argument("--lr_z", type=float, default=4e-4, help="Learning rate for z param")
    parser.add_argument("--mom_main", type=float, default=0.9, help="Momentum for main params")
    parser.add_argument("--mom_z", type=float, default=0.9, help="Momentum for z param")

    cfg = parser.parse_args()
    train(cfg)
