
# 🧠 Knapsack GFlowNet  
_Efficient, probabilistic 0‑1 Knapsack optimisation for Economists, OR folks and ML Engineers_

---

## 🚀 One‑Minute Pitch
The classic **0‑1 Knapsack** has 2^n feasible baskets.  
A **Generative Flow Network (GFlowNet)** learns a *probability distribution* over **all** baskets so that higher‑value sets are sampled more often.  
For *n = 15* the global optimum is typically found after **≈ 2 600** sampled trajectories (20 epochs × 128‑batch) instead of traversing all 32 768 leaves.

---

## 👥 Audience Cheat‑Sheet

| You are… | Read **first** | Why this repo is useful |
|----------|---------------|-------------------------|
| **🎓 Economist** (no heavy maths / AI) | `0_A-ECON.md` | Study the *whole* welfare distribution, not a single optimum |
| **🛠 OR / Optimisation** | `0_B-OR.md` | Think *Monte‑Carlo DP × Normalising Flows* – sample instead of enumerate |
| **🤖 ML / AI Engineer** | `0_C-ML.md` | Trajectory‑Balance loss, GPU batching, W&B sweeps, three state encodings |

> Each primer is two paragraphs; skip what you already know.

---

## 🗂️ Repository Layout

```
.
├── data/                       # Pickled toy instance  (u, t, B)
│   └── data.pickle
├── env/
│   └── environment.yaml        # Conda definition  (Python 3.13.2)
├── metrics/
│   └── probability_distribution_on_batch.py
├── models/                     # Three GFlowNet variants
│   ├── baseline_v1.py          # v1 – budget‑blind
│   ├── block_traj_v2.py        # v2 – online blocking
│   └── remaining_budget_v3.py  # v3 – dynamic budget  ✅ best
├── reward/
│   ├── analytical_reward.py    # Brute‑force oracle (≤20 items)
│   └── reward.py               # Batch reward computation
├── sweep/
│   └── sweep.yaml              # W&B Bayesian optimisation config
├── train.py                    # 🏁 Entry‑point
├── requirements.txt            # pip alternative to Conda
└── README.md                   # ← you are here
```

*(A local `wandb/` directory is created at run‑time; add it to `.gitignore`.)*

---

## 🔬 Algorithm in a Nutshell
1. **State** = `(-1, 0, 1)` code for each item + **remaining budget** (v3).  
2. **Policy** `π_θ(a | s)` queried for 128 states **in parallel**.  
3. **Trajectory‑Balance loss**  

   \[
   \mathcal L(τ)=\bigl[\log P_{θ}(τ)+\log Z-\log R(τ)\bigr]^2
   \]

   with a learnable scalar `log Z`.  
4. Gradient step → discard batch → sample a fresh one. Eventually every leaf is visited stochastically.

---

## 📈 Key Results



_Interactive dashboards on Weights & Biases project **gflownet‑knapsack**._

---

## ⚡ Quick‑Start

### 1 · Environment

```bash
git clone https://github.com/your‑handle/gflownet‑knapsack.git
cd gflownet‑knapsack

conda env create -f env/environment.yaml
conda activate gflownet-knapsack
# or  python -m pip install -r requirements.txt
```

### 2 · Train

```bash
python train.py --model_version v3 --num_epochs 1000 --batch_size 128
```

Dataset default: `data/data.pickle` – override with `--data_path`.

### 3 · Hyper‑parameter sweep (optional)

```bash
wandb sweep sweep/sweep.yaml
wandb agent <entity>/<project>/<sweep-id>
```

Objective = minimise **mean_kl_200_to_end**.

---

## 💾 Data Format

```python
{
  "u": torch.FloatTensor(n),  # utilities
  "t": torch.FloatTensor(n),  # costs
  "B": float                  # budget
}
```

---

## 🛠 Extensions
* **Bilevel optimisation** – reuse across similar instances.  
* **Transformer encoder** – scale beyond 25 items.  
* **Distributed training** – multi‑GPU / multi‑node.

PRs welcome!

---

## 📜 Licence
MIT

> _Made with ❤️ & a MacBook Air (M2) chilled on an ice‑pack._
