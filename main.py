import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import itertools
import matplotlib.pyplot as plt

from models import GFlowNet

with open('data.pickle', 'rb') as f:
    data = pickle.load(f)


num_items = torch.tensor(data['num_items'])
n_controlled = torch.tensor(data['n_controlled'])
u = torch.tensor(data['u'])
t = torch.tensor(data['t'])
B = torch.tensor(data['B'])
n_consumers = torch.tensor(data['n_consumers'])


def compute_reward(selected, u, t, B):
    """
    Calcule la récompense : sum((u - t) * x) si sum(t * x) <= B, sinon 0.
    - selected : (batch_size, num_items) - vecteur de sélection (-1 ou 1)
    - u : (num_items,) - utilités
    - t : (num_items,) - prix
    - B : budget (scalaire)
    """
    # Convertir selected en binaire : -1 -> 0, 1 -> 1
    selected_binary = torch.where(selected <= 0, torch.tensor(0.0), selected)  # (batch_size, num_items)

    # Calculer le coût total pour chaque trajectoire
    total_cost = (selected_binary * t).sum(dim=1)  # (batch_size,)

    # Calculer la récompense potentielle
    utility_minus_cost = u - t  # (num_items,)
    utility_minus_cost = utility_minus_cost.expand(selected.size(0), num_items).to(dtype=torch.float32)

    reward = (utility_minus_cost * selected_binary).sum(dim=1)  # (batch_size,)

    # Vérifier le budget : si total_cost > B, reward = 0
    reward = torch.where(total_cost > B, torch.tensor(0.0, dtype=torch.float32), reward)

    return reward


batch_size = 512
 
model = GFlowNet(num_items, batch_size = batch_size)

# Paramètres d'entraînement

# Définir les learning rates
lr_main = 0.00005  # Learning rate pour les autres paramètres
lr_z = 0.0015      # Learning rate différent pour model.z

# Séparer les paramètres
z_params = [model.z]  # model.z est un paramètre unique
other_params = [param for param in model.parameters() if param is not model.z]

# Configurer l'optimiseur avec deux groupes de paramètres
optimizer = torch.optim.Adam([
    {'params': other_params, 'lr': lr_main},  # Groupe pour les autres paramètres
    {'params': z_params, 'lr': lr_z}          # Groupe pour model.z
])

num_epochs = 3000  # Nombre d'itérations d'entraînement

# Calcul de Z analytique et CDF analytique
all_trajectories = torch.tensor(list(itertools.product([-1, 1], repeat=num_items)), dtype=torch.float32)

all_rewards = compute_reward(all_trajectories, u, t, B)
Z_analytic = all_rewards.sum().item()
p_tau_analytic = all_rewards / Z_analytic
sorted_rewards, indices = all_rewards.sort()
max_reward_analytic = all_rewards.max().item()
cdf_analytic = p_tau_analytic[indices].cumsum(dim=0)



# Initialiser les listes pour stocker les valeurs
losses = []
kl_divs = []
max_rewards_batch = []
epochs = []


realZ = torch.tensor(Z_analytic)

# Boucle d'entraînement
for epoch in range(num_epochs):
    optimizer.zero_grad()

    sequence_log_prob, selected = model.generate_trajectories(B, u, t, batch_size=batch_size)
    reward = compute_reward(selected, u, t, B)

    # Correction : utiliser model.z au lieu de reLz (non défini)
    log_z = model.z  # Suppose que model.z est le paramètre appris
    log_reward = torch.log(reward + 1e-6)  # Ajouter epsilon pour éviter log(0)
    diff = log_z + sequence_log_prob - log_reward

    # Limiter les valeurs pour éviter des explosions
    diff_clamped = torch.clamp(diff, min=-100.0, max=100.0)

    # Calculer la perte avec les valeurs limitées
    loss = torch.mean(diff_clamped ** 2)

    loss.backward()
    optimizer.step()

    # Stocker la loss
    losses.append(loss.item())

    # Stocker le max reward dans le batch
    max_reward_batch = reward.max().item()
    max_rewards_batch.append(max_reward_batch)

    sequence_prob = torch.exp(sequence_log_prob)
    p_tau_target = reward / Z_analytic
    p_tau_target = p_tau_target / p_tau_target.sum()
    sequence_prob = sequence_prob / sequence_prob.sum()
    kl_div = torch.sum(p_tau_target * torch.log(p_tau_target / (sequence_prob + 1e-6) + 1e-6)).item()
    
    kl_divs.append(kl_div)
    epochs.append(epoch + 1)
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Loss: {loss.item():.6f}")
    print(f"KL Divergence (échantillonnée): {kl_div:.6f}")
    print(f"Max Reward (batch): {max_reward_batch:.6f}")
    print(f"Probabilité (exemple): {sequence_prob[0].item():.6f}")
    print(f"Trajectoire (exemple): {selected[0]}")
    print(f"Récompense (exemple): {reward[0].item():.6f}")
    print(f"Z (modèle): {model.z.item():.6f}")
    print("-" * 50)

# Graphique avec trois sous-graphiques dans la même fenêtre (l'un au-dessus de l'autre)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

# Sous-graphique 1 : Loss
ax1.plot(epochs, losses, label='Loss', color='purple')
ax1.set_ylabel('Loss')
ax1.set_title('Évolution de la Loss')
ax1.legend()
ax1.grid(True)

# Sous-graphique 2 : KL Divergence
ax2.plot(epochs, kl_divs, label='KL Divergence', color='blue')
ax2.set_ylabel('KL Divergence')
ax2.set_title('Évolution de la Divergence KL')
ax2.legend()
ax2.grid(True)

# Sous-graphique 3 : Récompenses Max
ax3.plot(epochs, [max_reward_analytic] * len(epochs), label='Max Reward Analytique', color='red', linestyle='--')
ax3.plot(epochs, max_rewards_batch, label='Max Reward Batch', color='green')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Récompense')
ax3.set_title('Évolution des Récompenses Max')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()



# Calcul de Z analytique et CDF analytique
all_trajectories = torch.tensor(list(itertools.product([-1, 1], repeat=num_items)), dtype=torch.float32)

all_rewards = compute_reward(all_trajectories, u, t, B)
Z_analytic = all_rewards.sum().item()
p_tau_analytic = all_rewards / Z_analytic
sorted_rewards, indices = all_rewards.sort()
max_reward_analytic = all_rewards.max().item()
cdf_analytic = p_tau_analytic[indices].cumsum(dim=0)



# Ton code
batch_size_cdf = 512
q_tau_gfn = torch.zeros(len(all_trajectories), dtype=torch.float32)
for i in range(0, len(all_trajectories), batch_size_cdf):
    print(i)
    batch_trajectories = all_trajectories[i:i + batch_size_cdf]
    log_probs = model.compute_trajectory_log_prob_batch(batch_trajectories, B, u, t, batch_size=batch_size)
    q_tau_gfn[i:i + batch_size_cdf] = torch.exp(log_probs.detach())

print("all_rewards shape:", all_rewards.shape)
print("q_tau_gfn shape:", q_tau_gfn.shape)

sorted_rewards_gfn, indices_gfn = all_rewards.sort()

# Normalisation (déjà dans ton code)
q_tau_gfn = q_tau_gfn / q_tau_gfn.sum()

# Probabilités triées (non cumulées)
probs_gfn_sorted = q_tau_gfn[indices_gfn]  # Probabilités GFlowNet
probs_analytic_sorted = p_tau_analytic[indices]  # Probabilités analytiques

# CDF (déjà calculée)
cdf_gfn = q_tau_gfn[indices_gfn].cumsum(dim=0)
cdf_analytic = p_tau_analytic[indices].cumsum(dim=0)

# Graphique avec deux sous-graphiques
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Sous-graphique 1 : Distribution de probabilité (non cumulée)
ax1.plot(sorted_rewards_gfn.numpy(), probs_gfn_sorted.numpy(), label='Probabilité GFlowNet', color='orange')
ax1.plot(sorted_rewards.numpy(), probs_analytic_sorted.numpy(), label='Probabilité Analytique', color='blue')
ax1.set_ylabel('Probabilité')
ax1.set_title('Distribution de Probabilité : Analytique vs GFlowNet')
ax1.legend()
ax1.grid(True)

# Sous-graphique 2 : CDF
ax2.plot(sorted_rewards_gfn.numpy(), cdf_gfn.numpy(), label='CDF GFlowNet', color='orange')
ax2.plot(sorted_rewards.numpy(), cdf_analytic.numpy(), label='CDF Analytique', color='blue')
ax2.set_xlabel('Récompense')
ax2.set_ylabel('Probabilité Cumulative')
ax2.set_title('Comparaison des CDF : Analytique vs GFlowNet')
ax2.legend()
ax2.grid(True)

# Ajustement de l'espacement
plt.tight_layout()
plt.show()