import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def compute_sorted_probabilities(rewards: torch.Tensor,
                                 model_log_probs: torch.Tensor,
                                 analytic_probs: torch.Tensor):
    """
    Transforme log_probas → probas, normalise, trie par reward croissante et
    renvoie (sorted_rewards, p_model_sorted, p_analytic_sorted).
    """
    # logp → p et normalisation
    model_probs    = torch.exp(model_log_probs.detach())
    model_probs   = model_probs / model_probs.sum()
    analytic_probs = analytic_probs / analytic_probs.sum()

    # tri
    sorted_rewards, indices    = torch.sort(rewards)
    p_model_sorted      = model_probs[indices]
    p_analytic_sorted   = analytic_probs[indices]

    return sorted_rewards, p_model_sorted, p_analytic_sorted


@torch.no_grad()
def plot_probability_distribution(sorted_rewards: torch.Tensor,
                                  pdf_model: torch.Tensor,
                                  pdf_numeric: torch.Tensor,
                                  epoch: int,
                                  wandb):
    """
    Trace et log uniquement la PDF (non cumulée) pour un epoch donné.
    """
    # Conversion en numpy
    sr = sorted_rewards.cpu().numpy()
    pm = pdf_model.cpu().numpy()
    pa = pdf_numeric.cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sr, pm, label="GFlowNet PDF",   color="orange", marker='.', linestyle='-')
    ax.plot(sr, pa, label="Analytic PDF",   color="blue",   marker='x', linestyle='-')
    ax.set_xlabel("Reward")
    ax.set_ylabel("Probabilité")
    ax.set_title(f"Epoch {epoch} — PDF seule")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    wandb.log({"prob_distribution": wandb.Image(plt)}, step=epoch)
    plt.close()


@torch.no_grad()
def plot_cdf_distribution(sorted_rewards: torch.Tensor,
                          pdf_model: torch.Tensor,
                          pdf_analytic: torch.Tensor,
                          epoch: int,
                          wandb):
    """
    Trace et log uniquement la CDF (cumulative) pour un epoch donné.
    """
    # calcul CDF
    cdf_model    = pdf_model.cumsum(dim=0)
    cdf_analytic = pdf_analytic.cumsum(dim=0)

    # Conversion en numpy
    sr = sorted_rewards.cpu().numpy()
    cm = cdf_model.cpu().numpy()
    ca = cdf_analytic.cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sr, cm, label="GFlowNet CDF",   color="orange", marker='.', linestyle='-')
    ax.plot(sr, ca, label="Analytic CDF",   color="blue",   marker='x', linestyle='-')
    ax.set_xlabel("Reward")
    ax.set_ylabel("Probabilité Cumulative")
    ax.set_title(f"Epoch {epoch} — CDF seule")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    wandb.log({"cdf_distribution": wandb.Image(plt)}, step=epoch)
    plt.close()


@torch.no_grad()
def log_distribution_and_cdf(epoch: int,
                              rewards: torch.Tensor,
                              model_log_probs: torch.Tensor,
                              analytic_probs: torch.Tensor,
                              wandb,
                              step_interval: int = 50):
    """
    Fonction principale : à chaque `step_interval` epochs, calcule
    une seule fois les PDF triées puis appelle les deux tracés ci‑dessus.
    """
    if epoch % step_interval != 0:
        return

    # 1) calcul des vecteurs triés
    sorted_rewards, pdf_model, pdf_analytic = compute_sorted_probabilities(
        rewards, model_log_probs, analytic_probs
    )

    # 2) affichage / log
    plot_probability_distribution(sorted_rewards, pdf_model, pdf_analytic, epoch, wandb)
    plot_cdf_distribution(      sorted_rewards, pdf_model, pdf_analytic, epoch, wandb)