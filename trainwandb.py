"""
train.py

Script d'entraînement pour GFlowNet Knapsack (versions baseline v1 & block_traj v2)
----------------------------------------------------------------------------------
Cette version gère deux variantes de GFlowNet :
  - v1 (baseline) : génère toutes les séquences sans contrainte explicite de budget.
  - v2 (block_traj) : stoppe la génération dès que le budget restant est négatif,
                      et remplit les décisions restantes avec 0 pour éviter les
                      trajectoires invalides.

Fonctionnalités clés :
  - Choix du modèle via l'argument --model_version (v1 ou v2).
  - Chargement des données (u, t, B) depuis un fichier pickle.
  - Entraînement en SGD à momentum, avec le paramètre z appris séparément.
  - Logging temps-réel des métriques (loss, reward moyen, reward max, KL divergence)
    dans Weights & Biases.
  - Barre de progression TQDM affichant live Loss, Rew_Avg, Rew_Max, KL_Div.
  - Détection automatique du device (GPU > MPS > CPU).

Usage exemple :
    python train.py --model_version v2 --data_path data.pickle --batch_size 512 --num_epochs 200
"""


import argparse
import wandb
import torch
import pickle
from tqdm import tqdm


from reward.reward import compute_reward
from reward.analytical_reward import compute_analytical_reward

from metrics.probability_distribution_on_batch import log_distribution_and_cdf

from models.baseline_v1 import GFlowNet as Baseline
from models.block_traj_v2 import GFlowNet as Block_traj
from models.remaining_budget_v3 import GFlowNet as Dynamic_budget


# Mapping des versions de modèles
MODEL_MAP = {
    "v1": Baseline,
    "v2": Block_traj,
    "v3": Dynamic_budget,
}

def train(args):
    # Initialise un run W&B
    run_name = f"{args.model_version}_bs{args.batch_size}_ep{args.num_epochs}_{wandb.util.generate_id()[:5]}"
    wandb.init(project="gflownet-knapsack", name=run_name, config=vars(args))
    cfg = wandb.config

    #configure batch_size
    batch_size = cfg.batch_size

    # Sélection du device
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")
    device_str = "GPU" if device.type == "cuda" else "MPS" if device.type == "mps" else "CPU"
    print(f"Training on {device_str}")

    # Charge les données
    with open(cfg.data_path, "rb") as f:
        data = pickle.load(f)

    # calcul analytique avant training
    max_reward_bruteforce = compute_analytical_reward(
        u_vals = data["u"],
        t_vals = data["t"],
        B_val  = data["B"]
    )
    print(f"🎯 Reward max analytique (bruteforce) = {max_reward_bruteforce:.4f}")
    wandb.config.update({"max_reward_bruteforce": max_reward_bruteforce})


    # Tensor modification, converting to (batch_size, num_item) in float32
    u = torch.tensor(data["u"], dtype=torch.float32, device=device).expand(batch_size, -1).detach()
    t = torch.tensor(data["t"], dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1).detach()
    Budget = torch.tensor(data["B"], dtype=torch.float32, device=device).view(1, 1).expand(batch_size, 1).detach()
    num_items = u.size(1)

    # Instancie le modèle selon la version choisie
    ModelClass = MODEL_MAP[cfg.model_version]
    model = ModelClass(num_items=num_items, embedding_dim=cfg.embedding_dim, hidden_dim=cfg.hidden_dim, init_value_z=cfg.init_value_z).to(device)

    # Sépare les groupes de paramètres
    z_params = [model.z]
    other_params = [p for p in model.parameters() if p is not model.z]

    # Optimiseur SGD à momentum
    optimizer = torch.optim.SGD([
    {"params": other_params, "lr": cfg.lr_main, "momentum": cfg.mom_main, "weight_decay": 1e-4},
    {"params": z_params,     "lr": cfg.lr_z,    "momentum": cfg.mom_z,    "weight_decay": 0.0},  # PAS de decay sur Z
    ])

    old_max_reward = torch.tensor(0)
    kl_values = []  # Liste pour stocker les KL à partir de l'époque 200
    # Boucle d'entraînement
    # Barre de progression
    pbar = tqdm(range(cfg.num_epochs),
                desc=f"Modèle: {cfg.model_version} | Device: {device_str}")
    for epoch in pbar:
        optimizer.zero_grad()
        seq_logp, selected = model.generate_trajectories(Budget, u, t, batch_size, num_items, device)

        # Vérification rapide
        assert seq_logp.requires_grad, "❌ sequence_log_prob n'est pas différentiable !"

        reward = compute_reward(selected, u, t, Budget, num_items)

        # Utilisation de la méthode compute_loss du modèle
        loss = model.compute_loss(seq_logp, reward)
        loss.backward()
        optimizer.step()

        # Calcul KL divergence et max reward
        model.eval()
        with torch.no_grad():
            avg_r = reward.mean().item()
            prob_seq = torch.exp(seq_logp)
            
            p_target = reward / reward.sum()
            prob_seq = prob_seq / prob_seq.sum()
            
            kl = (p_target * (torch.log(p_target + 1e-6)
                              - torch.log(prob_seq + 1e-6))).sum().item()
            max_r = reward.max().item()
            if max_r > old_max_reward:
                old_max_reward = max_r
            
            # Stocke KL à partir de l'époque 200
            if epoch >= 200:
                kl_values.append(kl)
            
            #Numérical CDF from GFlowNet sampler, data points = batch_size
            log_distribution_and_cdf(epoch,
                              reward,         # torch.Tensor (N,)
                              seq_logp,       # log-probs modèle (N,)
                              p_target, # probs analytiques (N,)
                              wandb,
                              step_interval=50)
        model.train()



        # Mise à jour de la barre de progression
        pbar.set_postfix({
            "Loss":   f"{loss.item():.4f}",
            "Rew_Avg": f"{avg_r:.2f}",
            "Rew_Max": f"{max_r:.2f}",
            "KL_Div":  f"{kl:.4f}" 
        })
        # Log des métriques à W&B
        wandb.log({
            "loss":       loss.item(),
            "avg_reward": avg_r,
            "kl_div":     kl,
            "max_reward": old_max_reward,
            "z":          model.z.item(),
        }, step=epoch)
        
    if kl_values:
        mean_kl = sum(kl_values) / len(kl_values)
        if mean_kl == 0 or mean_kl is None:
            penalized_kl = 2.0  # Suppose que ta moyenne normale est 0.2, 2.0 c’est soft mais clair.
            wandb.log({
                "mean_kl_200_to_end": penalized_kl,
                "kl_zero": True
            })
        else:
            wandb.log({"mean_kl_200_to_end": mean_kl})
            print(f"Moyenne KL (époques 200 à {len(kl_values) + 199}) : {mean_kl}")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GFlowNet knapsack with wandb + SGD")
    parser.add_argument("--model_version", choices=list(MODEL_MAP.keys()), default="v3")
    parser.add_argument("--data_path",    type=str,   default="data.pickle")
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--num_epochs",   type=int,   default=400)
    parser.add_argument("--lr_main",      type=float, default=2e-3)
    parser.add_argument("--lr_z",         type=float, default=4e-4)
    parser.add_argument("--embedding_dim",         type=int, default=150)
    parser.add_argument("--hidden_dim",         type=int, default=360)
    parser.add_argument("--init_value_z", type=float, default=12)
    parser.add_argument("--mom_main", type=float, default=0.9)
    parser.add_argument("--mom_z", type=float, default=0.9)
    args = parser.parse_args()
    train(args)
