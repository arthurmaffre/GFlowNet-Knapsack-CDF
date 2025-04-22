# metrics.py
import matplotlib.pyplot as plt
import torch

@torch.no_grad()
def log_distribution_and_cdf(epoch: int,
                              rewards: torch.Tensor,        # shape (N,)
                              model_log_probs: torch.Tensor,# shape (N,)
                              analytic_probs: torch.Tensor, # shape (N,)
                              wandb,
                              step_interval: int = 50):
    """
    À chaque `step_interval` epochs, trace et log dans W&B :
     - en haut : PDF (prob model vs prob analytique)
     - en bas  : CDF (model vs analytique)
    """
    if epoch % step_interval != 0:
        return

    # 1) transforme en probabilités et normalise
    model_probs    = torch.exp(model_log_probs.detach())
    model_probs   /= model_probs.sum()
    analytic_probs = analytic_probs / analytic_probs.sum()

    # 2) trie par reward croissante
    sorted_rewards, indices = torch.sort(rewards)
    p_model_sorted      = model_probs[indices]
    p_analytic_sorted   = analytic_probs[indices]

    # 3) calcule CDF
    cdf_model    = p_model_sorted.cumsum(dim=0)
    cdf_analytic = p_analytic_sorted.cumsum(dim=0)

    # 4) conversion numpy
    sr_np    = sorted_rewards.cpu().numpy()
    pm_np    = p_model_sorted.cpu().numpy()
    pa_np    = p_analytic_sorted.cpu().numpy()
    cm_np    = cdf_model.cpu().numpy()
    ca_np    = cdf_analytic.cpu().numpy()

    # 5) trace en deux sous‑graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # PDF
    ax1.plot(sr_np, pm_np, label="GFlowNet PDF",   marker='.', linestyle='-')
    ax1.plot(sr_np, pa_np, label="Analytic PDF",    marker='x', linestyle='-')
    ax1.set_ylabel("Probabilité")
    ax1.set_title(f"Epoch {epoch} — Distribution de Probabilité")
    ax1.legend()
    ax1.grid(True)

    # CDF
    ax2.plot(sr_np, cm_np, label="GFlowNet CDF",    marker='.', linestyle='-')
    ax2.plot(sr_np, ca_np, label="Analytic CDF",    marker='x', linestyle='-')
    ax2.set_xlabel("Reward")
    ax2.set_ylabel("Probabilité Cumulative")
    ax2.set_title(f"Epoch {epoch} — CDF")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    wandb.log({"dist_and_cdf": wandb.Image(plt)}, step=epoch)
    plt.close()