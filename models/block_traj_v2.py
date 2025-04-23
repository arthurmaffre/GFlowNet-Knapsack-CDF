from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor






class GFlowNet(nn.Module):
    """
        Version 3 du GFlowNet : amélioration de la version 2 en passant le *budget restant* 
        (remaining_budget) à chaque étape du forward, plutôt qu'un budget statique initial.

        Objectif :
        -----------
        - Augmenter l'efficacité du modèle en conditionnant les décisions non seulement 
        sur le budget initial mais aussi sur l'état réel de consommation du budget à 
        chaque étape de la trajectoire.
        - Cela permet d'éliminer plus tôt les trajectoires impossibles et d'apprendre 
        une politique plus précise, en respectant dynamiquement les contraintes de budget.

        Implémentation :
        -----------------
        - Le `remaining_budget` sera mis à jour étape par étape dans `generate_trajectories`.
        - Le forward du modèle prendra en entrée ce `remaining_budget` actualisé pour améliorer
        la qualité des probabilités d'actions calculées.
"""
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 150,
        hidden_dim: int = 360,
        init_value_z: float = 12.0,
    ) -> None:
        super().__init__()

        self.num_items = num_items
        self.embed_sel = nn.Linear(num_items, embedding_dim)   # x ∈ {-1,1}ⁿ
        self.embed_B   = nn.Linear(1,           embedding_dim) # remaining budget
        self.embed_u   = nn.Linear(num_items,   embedding_dim) # utilities
        self.embed_t   = nn.Linear(num_items,   embedding_dim) # prices

        # 4-layer MLP : d → d → d/2 → d/4 → d/8 → 1
        dims = [4 * embedding_dim, hidden_dim,
                hidden_dim // 2, hidden_dim // 4, hidden_dim // 8]
        mlp = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            mlp += [nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.ReLU()]
        self.mlp_stack = nn.ModuleList(mlp)
        self.head      = nn.Linear(dims[-1], 1)  # produces *logit*

        # log-partition parameter log Z  (learned scalar)
        self.log_z = nn.Parameter(torch.tensor(init_value_z, dtype=torch.float32))

        # sane init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # FORWARD  (pure scorer – no sampling, keeps gradients clean)
    # ------------------------------------------------------------------
    def forward(
        self,
        selected: Tensor,          # (B, n)  values in {-1, 1}
        remaining_B: Tensor,       # (B, 1)
        u: Tensor,                 # (B, n)
        t: Tensor,                 # (B, n)
    ) -> Tensor:                   # (B, 1)  Bernoulli probability
        sel_emb = F.relu(self.embed_sel(selected))
        B_emb   = F.relu(self.embed_B(remaining_B))
        u_emb   = F.relu(self.embed_u(u))
        t_emb   = F.relu(self.embed_t(t))

        h = torch.cat((sel_emb, B_emb, u_emb, t_emb), dim=1)
        for layer in self.mlp_stack:
            h = layer(h)           # ReLU already in stack
        logits = self.head(h)
        return torch.sigmoid(logits)  # Bernoulli *p*

    # ------------------------------------------------------------------
    # TRAJECTORY GENERATION  (stochastic, differentiable)
    # ------------------------------------------------------------------
    def generate_trajectories(
        self,
        B_init: Tensor,            # (B, 1)
        u: Tensor,                 # (B, n)
        t: Tensor,                 # (B, n)
        batch_size: int,
        num_items: int,
        device
    ) -> tuple[Tensor, Tensor]:
        """
        Sample a batch of trajectories and accumulate their log-probabilities.

        Returns
        -------
        sequence_logp : Tensor, shape (B,)
            log P_θ(τ) for each generated trajectory.
        selected      : Tensor, shape (B, n)
            Final decision vectors in {-1, 1}.
        """
        
        device = u.device

        # static embeddings (utilities, prices) are the same at every step
        u_emb_fixed = F.relu(self.embed_u(u))
        t_emb_fixed = F.relu(self.embed_t(t))

        selected = torch.full((batch_size, num_items), -1.0, device=device)
        logp_acc = torch.zeros(batch_size, device=device)
        remaining_B = B_init.clone()                # mutable (B,1)

        for i in range(num_items):
            # ---- forward (uses current decisions) --------------------------
            sel_emb = F.relu(self.embed_sel(selected))
            B_emb   = F.relu(self.embed_B(B_init))
            h = torch.cat((sel_emb, B_emb, u_emb_fixed, t_emb_fixed), dim=1)
            for layer in self.mlp_stack:
                h = layer(h)
            probs = torch.sigmoid(self.head(h)).squeeze(-1)

            feasible = remaining_B.squeeze(1) >= t[:, i]
            probs = probs * feasible.float()
            dist  = torch.distributions.Bernoulli(probs=probs.clamp(1e-8, 1-1e-8))

            act = dist.sample()                       # 0 / 1
            logp_acc = logp_acc + dist.log_prob(act)

            # ---- safe update (out-of-place) --------------------------------
            new_selected = selected.clone()           # version bump → autograd safe
            new_selected[:, i] = act * 2.0 - 1.0      # map {0,1}→{-1,1}
            selected = new_selected

            remaining_B = remaining_B - (act * t[:, i]).unsqueeze(1)

        return logp_acc, selected

    def compute_trajectory_log_prob_batch(self,
                                          trajectories: torch.Tensor,
                                          budget: float,
                                          u:       torch.Tensor,
                                          t:       torch.Tensor,
                                          batch_size: int = None) -> torch.Tensor:
        """
        Calcule log P(tau) pour un batch de trajectoires prédéfinies
        """
        batch_size = trajectories.size(0)
        num_items  = trajectories.size(1)

        selected      = torch.zeros(batch_size, num_items)
        budget_tensor = torch.full((batch_size, 1), budget)
        u_expand      = u.unsqueeze(0).expand(batch_size, -1).float()
        t_expand      = t.unsqueeze(0).expand(batch_size, -1).float()
        log_prob      = torch.zeros(batch_size)

        for i in range(num_items):
            action_prob = self.forward(selected, budget_tensor, u_expand, t_expand)
            dist        = torch.distributions.Bernoulli(probs=action_prob)

            # Convertir -1/1 → 0/1 pour Bernoulli
            actions = (trajectories[:, i] + 1) / 2
            actions = actions.unsqueeze(-1)

            log_prob += dist.log_prob(actions).squeeze(-1)
            selected[:, i] = trajectories[:, i]

        return log_prob

    def compute_loss(self,
                     sequence_log_prob: torch.Tensor,
                     reward:            torch.Tensor) -> torch.Tensor:
        """
        MSE Loss sur (log_z + log π(a) – log r), clamp pour stabilité

        Args:
            sequence_log_prob : Tensor(batch_size,)
            selected          : Tensor(batch_size, num_items)
            reward            : Tensor(batch_size,)
        Returns:
            loss : scalar
        """
        z_param = self.log_z
        log_z = z_param.unsqueeze(0)

        # log de la reward(+eps)
        log_r = torch.log(reward.clamp(min=1e-6))
        # diff clampée

        diff  = torch.clamp(log_z + sequence_log_prob - log_r,
                            min=-100.0, max=100.0)
        return diff.square().mean()