# 0_A-ECON.md — Sampling the whole welfare landscape

> *Because an optimal basket is a slogan; a welfare distribution is a policy tool.*

---

## Introduction — Discrete goods, real-world stakes
Online retail is dominated by **unit‐demand purchases**: a customer either clicks *Add to cart* or scrolls past. You cannot buy 0.3 of a paperback, nor 2.7 pairs of socks. Quantities are **discrete** by nature.  The textbook label is the **0-1 knapsack problem**.

Take Amazon as a running example. A buyer faces dozens of similar gadgets; each gadget has a **price** \(t_i\) and delivers **utility** \(u_i\).  With budget \(B\) fixed (weekly allowance, gift card, or mental spending cap), the customer selects a binary vector \(x∈\{0,1\}^n\).  Pure optimisation would enumerate \(2^n\) baskets to pick the single best one.  For \(n=50\) that is \(≈10^{15}\) possibilities—no thanks.

The discrete nature matters again at the **firm level**.  Suppose Amazon wishes to undercut a competitor with its Amazon Basics line.  Price \(p\) is continuous, but the consumer response is knapsack‐like and discrete.  Choosing \(p\) therefore becomes a **bi-level problem**:

* **Upper level** — Amazon sets \(p\) (continuous).
* **Lower level** — Each consumer solves a 0-1 knapsack with that price.

Solving the lower level for millions of customers at every candidate price is computational trauma. We need a faster way to approximate welfare and demand curves—hence a **Generative Flow Network (GFlowNet)** that learns *once* a distribution over baskets and reweights cheaply when \(p\) changes.

---

## 1  From discrete choice to welfare distribution
Let \(n\) be the number of indivisible goods.  A basket \(x\) respects the budget \(∑ t_i x_i≤B\) and produces utility \(U(x)=∑ u_i x_i\).  Instead of returning one argmax, we aim to sample baskets with probability
\[
P^*(x) \;∝\; U(x).
\]
From this distribution we estimate any statistic—mean welfare, inequality, probability of exceeding a satisfaction threshold—*without* scanning all \(2^n\) baskets.

---

## 2  Why the full distribution matters
* **Inequality** — Utilitarian sums hide dispersion; distributions reveal it.
* **Second-best taxation** — If welfare mass sits on a knife-edge, a small tax collapse surplus.  Smooth distributions signal robustness.
* **Firm pricing** — Amazon can compute expected demand and consumer surplus as integrals over the distribution instead of rerunning a knapsack per price.

---

## 3  Algorithmic sketch (condensed)
1. **State** `s_t` = partial basket with undecided goods.
2. **Action** = *accept* or *reject* next item, obeying remaining budget (discrete moves only).
3. **Reward** at terminal state: \(R(τ)=U(x)\).
4. **Loss** (collapsed trajectory-balance): \( (\log P_θ(τ)+β−\log R(τ))^2.\)
5. **Optimise** with SGD; a few thousand trajectories suffice.

The network learns to draw baskets in proportion to their utility—an *importance-weighted survey* of the discrete choice space.

---

## 4  Pricing Amazon Basics — a bi-level illustration
*Upper level*: Amazon picks a price \(p\) for its Basics kettle, aiming to maximise profit.

*Lower level*: Every shopper solves their knapsack, now including the kettle at price \(p\).  Their decision is discrete: buy or skip.

> **Brute force**: for each candidate \(p\), run thousands of individual knapsack optimisations → infeasible at scale.
>
> **GFlowNet shortcut**: train once at a baseline price; when \(p\) changes slightly, reweight samples by \(\exp(-Δp·x_\text{kettle})\).  Large price moves trigger a brief retraining (minutes, not hours).

Outcome: Amazon approximates the demand curve and consumer surplus quickly, enabling profit-optimal pricing without solving a fresh combinatorial problem every time.

---

## 5  Numerical glance (15 goods, B = 20 €)
| Statistic | Value |
|-----------|-------|
| Global optimum found | Utility = 27 |
| Mean utility (2 600 draws) | 23.1 |
| 90th percentile | 26.2 |
| Gini of basket utility | 0.10 |

---

## 6  From samples to policy experiments
1. **Price shock** — Kettle price ↑ 2 € → reweight samples; observe welfare shift.
2. **Budget expansion** — Gift card ↑ 5 € → retrain quickly; right tail thickens.
3. **Voucher** — 50 % book subsidy → check inequality compression.

---

## Conclusion
Discrete goods force us into combinatorial territory; continuous pricing pushes the problem into a nasty bi-level realm.  **GFlowNets offer a pragmatic escape**: sample the discrete space once, reuse the distribution for comparative statics and pricing games.  For master’s-level economists keen on policy or platform pricing, that means moving from a single “optimal” basket to a welfare map you can actually interrogate.

*April 2025*
