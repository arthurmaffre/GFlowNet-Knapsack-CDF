import torch
import torch.nn as nn
import torch.nn.functional as F







class GFlowNet(nn.Module):
    """
    GFlowNet baseline version 1

    Args:
        num_items     : nombre d'items à choisir
        batch_size    : taille du batch
        embedding_dim : dimension des embeddings
        hidden_dim    : dimension de la couche cachée initiale
    """
    def __init__(self,
                 num_items: int,
                 embedding_dim: int = 150,
                 hidden_dim: int = 360,
                 init_value_z: float = 12):
        super().__init__()
        self.num_items     = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim

        # Embeddings linéaires
        self.selected_embedding = nn.Linear(num_items, embedding_dim)
        self.budget_embedding   = nn.Linear(1, embedding_dim)
        self.u_embedding        = nn.Linear(num_items, embedding_dim)
        self.t_embedding        = nn.Linear(num_items, embedding_dim)

        # Couches cachées dynamiques
        self.fc1   = nn.Linear(4 * embedding_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.fc2   = nn.Linear(hidden_dim, hidden_dim // 2)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)

        self.fc3   = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.norm3 = nn.LayerNorm(hidden_dim // 4)

        self.fc4   = nn.Linear(hidden_dim // 4, hidden_dim // 8)
        self.norm4 = nn.LayerNorm(hidden_dim // 8)

        # Couche de sortie log-probs
        self.fc5   = nn.Linear(hidden_dim // 8, 1)

        # Paramètre appris pour la partition function Z
        self.z     = nn.Parameter(torch.tensor(init_value_z, dtype=torch.float32))

    def forward(self,
                selected: torch.Tensor,
                budget:   torch.Tensor,
                u:        torch.Tensor,
                t:        torch.Tensor) -> torch.Tensor:
        """
        Forward pass pour calculer la probabilité d'action

        Args:
            selected : Tensor(batch_size, num_items), -1 ou 1
            budget   : Tensor(batch_size, 1)
            u        : Tensor(batch_size, num_items)
            t        : Tensor(batch_size, num_items)

        Returns:
            action_probs : Tensor(batch_size, 1), probabilité (sigmoid)
        """
        # Embedding + activation
        selected_emb = F.relu(self.selected_embedding(selected))
        budget_emb   = F.relu(self.budget_embedding(budget))
        u_emb        = F.relu(self.u_embedding(u))
        t_emb        = F.relu(self.t_embedding(t))

        # Concaténation des embeddings\        
        combined = torch.cat([selected_emb, budget_emb, u_emb, t_emb], dim=1)

        # Couches cachées avec normalisation
        x = F.relu(self.norm1(self.fc1(combined)))
        x = F.relu(self.norm2(self.fc2(x)))
        x = F.relu(self.norm3(self.fc3(x)))
        x = F.relu(self.norm4(self.fc4(x)))

        # Logits et sigmoid
        action_scores = self.fc5(x)
        return torch.sigmoid(action_scores)

    def generate_trajectories(self,
                              Budget: torch.Tensor,
                              u:      torch.Tensor,
                              t:      torch.Tensor,
                              batch_size: int,
                              num_items: int,
                              device: torch.device) -> (torch.Tensor, torch.Tensor):
        """
        Génère un batch de trajectoires en choisissant séquentiellement chaque item

        Returns:
            sequence_log_prob : Tensor(batch_size,), log P(tau)
            selected          : Tensor(batch_size, num_items), décisions -1/1
        """
        

        selected = torch.zeros(batch_size, num_items, device=device)
        sequence_log_prob = torch.zeros(batch_size, device=device)

        # Boucle sur les items
        for i in range(num_items):
            # Calcul de la probabilité de prendre l'item i
            action_prob = self.forward(selected, Budget, u, t)
            dist        = torch.distributions.Bernoulli(probs=action_prob)
            

            # Échantillonnage et log-prob
            action = dist.sample()                        # 0/1
            logp   = dist.log_prob(action).squeeze(-1)    # (batch_size,)

            # Update selected: 1 -> +1, 0 -> -1
            selected = selected.clone()
            selected[:, i] = torch.where(action.squeeze()==1, 1.0, -1.0)
            sequence_log_prob = sequence_log_prob + logp

        return sequence_log_prob, selected

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
        z_param = self.z
        log_z = z_param.unsqueeze(0)

        # log de la reward(+eps)
        log_r = torch.log(reward.clamp(min=14))
        # diff clampée

        diff  = torch.clamp(log_z + sequence_log_prob - log_r,
                            min=-100.0, max=100.0)
        return (diff ** 2).mean()


# models/block_traj_v2.py
import torch
from models.baseline_v1 import GFlowNet as BaseGFlowNet