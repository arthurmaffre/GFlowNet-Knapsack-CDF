import torch
import torch.nn as nn
import torch.nn.functional as F







class GFlowNet(nn.Module):
    def __init__(self, num_items, batch_size, embedding_dim=128):
        super(GFlowNet, self).__init__()
        self.batch_size = batch_size
        
        # Embeddings raisonnables
        self.selected_embedding = nn.Linear(num_items, embedding_dim)
        self.budget_embedding = nn.Linear(1, embedding_dim)
        self.u_embedding = nn.Linear(num_items, embedding_dim)
        self.t_embedding = nn.Linear(num_items, embedding_dim)
        
        # Réseau avec une couche supplémentaire
        self.fc1 = nn.Linear(4 * embedding_dim, 512)  # 4*128=512 -> 256
        self.norm1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 190)                # 256 -> 190
        self.norm2 = nn.LayerNorm(190)
        self.fc3 = nn.Linear(190, 90)                 # 190 -> 90
        self.norm3 = nn.LayerNorm(90)
        self.fc4 = nn.Linear(90, 45)                  # Nouvelle couche : 90 -> 45
        self.norm4 = nn.LayerNorm(45)                 # Normalisation pour la nouvelle couche
        self.fc5 = nn.Linear(45, 1)                   # Sortie : 45 -> 1 (ancienne fc4 renommée fc5)
        
        # Paramètre appris Z
        self.z = nn.Parameter(torch.tensor(13.8, dtype=torch.float32), requires_grad=True)

    def forward(self, selected, budget, u, t):
        """
        Entrées :
        - selected : (batch_size, num_items) - produits déjà sélectionnés
        - budget : (batch_size, 1) - budget restant
        - u : (batch_size, num_items) - utilités des produits
        - t : (batch_size, num_items) - prix des produits
        Sortie :
        - action_probs : (batch_size, 1) - probabilité d'ajouter l'item
        """
        # Calcul des embeddings
        selected_emb = F.relu(self.selected_embedding(selected))  # (batch_size, embedding_dim)
        budget_emb = F.relu(self.budget_embedding(budget))        # (batch_size, embedding_dim)
        u_emb = F.relu(self.u_embedding(u))                       # (batch_size, embedding_dim)
        t_emb = F.relu(self.t_embedding(t))                       # (batch_size, embedding_dim)
        
        # Concaténation des embeddings
        combined = torch.cat([selected_emb, budget_emb, u_emb, t_emb], dim=1)  # (batch_size, 4 * embedding_dim)
        
        # Passage dans les couches fully connected
        x = F.relu(self.norm1(self.fc1(combined)))  # 512 -> 256
        x = F.relu(self.norm2(self.fc2(x)))         # 256 -> 190
        x = F.relu(self.norm3(self.fc3(x)))         # 190 -> 90
        x = F.relu(self.norm4(self.fc4(x)))         # Nouvelle couche : 90 -> 45
        action_scores = self.fc5(x)                 # 45 -> 1
        
        # Probabilité d'ajouter avec sigmoid
        action_probs = torch.sigmoid(action_scores)
        return action_probs
    
    def generate_trajectories(self, budget, u, t, batch_size=16):
        num_items = u.size(1)
        selected = torch.zeros(batch_size, num_items, dtype=torch.float32, requires_grad=True)
        budget = torch.tensor([[budget]], dtype=torch.float32).expand(batch_size, 1)
        u = u.expand(batch_size, num_items).to(dtype=torch.float32)
        t = t.expand(batch_size, num_items).to(dtype=torch.float32)

        sequence_log_prob = torch.zeros(batch_size, dtype=torch.float32).unsqueeze(-1)

        for i in range(num_items):
            action_prob = self.forward(selected, budget, u, t)  # (batch_size, 1)
            action_dist = torch.distributions.Bernoulli(probs=action_prob)
            action = action_dist.sample()  # (batch_size,)
            action_logprob = action_dist.log_prob(action)

            selected = selected.clone()
            selected[:, i] = torch.where(action.squeeze() == 1, 1.0, -1.0)
            sequence_log_prob += action_logprob

        return sequence_log_prob.squeeze(-1), selected
    

    def compute_trajectory_log_prob_batch(self, trajectories, budget, u, t, batch_size=32):
        """
        Version batchée.
        Calcule la log-probabilité totale pour un batch de trajectoires.
        - trajectories : (batch_size, num_items) - vecteur de décisions (-1 ou 1)
        - budget : float - budget initial
        - u : (num_items,) - utilités
        - t : (num_items,) - prix
        Retourne : log P(tau) pour chaque trajectoire dans le batch (batch_size,)
        """
        num_items = u.size(0) if u.dim() == 1 else u.size(1)
        selected = torch.zeros(batch_size, num_items, dtype=torch.float32)
        budget_tensor = torch.tensor([[budget]], dtype=torch.float32).expand(batch_size, 1)
        u = u.to(torch.float32).expand(batch_size, num_items)
        t = t.to(torch.float32).expand(batch_size, num_items)
        
        log_prob = torch.zeros(batch_size, dtype=torch.float32)
        
        for i in range(num_items):
            print(selected.shape)
            
            # Calculer la probabilité d'action avec le modèle
            action_prob = self.forward(selected, budget_tensor, u, t)  # (batch_size, 1)
            action_dist = torch.distributions.Bernoulli(probs=action_prob)
            
            # Transformer les actions de -1/1 à 0/1 pour Bernoulli
            actions = (trajectories[:, i] + 1) / 2  # -1 -> 0, 1 -> 1
            actions = actions.unsqueeze(-1)  # (batch_size, 1) pour aligner avec action_prob
            
            # Ajouter la log-probabilité de l'action courante
            log_prob += action_dist.log_prob(actions).squeeze(-1)  # (batch_size,)
            
            # Mettre à jour l'état des items sélectionnés
            selected[:, i] = trajectories[:, i]  # Garde -1/1 pour self.forward si nécessaire
        
        print(log_prob)
        print(torch.mean(torch.exp(log_prob)))
        print(torch.max(torch.exp(log_prob)))
        return log_prob

