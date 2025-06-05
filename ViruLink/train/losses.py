import torch
import torch.nn.functional as F

def interval_ce(logits, bounds):
    p = F.softmax(logits, dim=1)
    m = torch.zeros_like(p)
    for i, (lo, up) in enumerate(bounds.tolist()):
        m[i, lo:up+1] = 1.0
    return -(torch.clamp((p * m).sum(1), 1e-12).log()).mean()

def exp_rank(logits, K_CLASSES):
    return (F.softmax(logits, 1) * torch.arange(K_CLASSES, device=logits.device)).sum(1)

def hinge_sq(pred, lo, up):
    return (torch.relu(lo - pred)**2 + torch.relu(pred - up)**2)

def tri_dual(r1, r2, lb, ub):
    l1 = hinge_sq(r1, torch.minimum(r2, lb.float()), torch.minimum(r2, ub.float()))
    l2 = hinge_sq(r2, torch.minimum(r1, lb.float()), torch.minimum(r1, ub.float()))
    return (l1 + l2).mean()

def CE_MSE_TRE_LOSS(la, lh, b, K_CLASSES, LAMBDA_INT, LAMBDA_TRI):
    ce  = interval_ce(la, b["lqa"]) + interval_ce(lh, b["lqh"])
    tri = tri_dual(exp_rank(la, K_CLASSES), exp_rank(lh, K_CLASSES), b["lrr"][:,0], b["lrr"][:,1])
    return LAMBDA_INT * ce + LAMBDA_TRI * tri

def cum_interval_bce(logits: torch.Tensor,
                     bounds: torch.Tensor) -> torch.Tensor:
    """
    Cumulative-link BCE with interval censoring.

    logits : shape [B, K] – raw logits for P(rank ≥ k)
    bounds : shape [B, 2] – (lower, upper) inclusive interval
    """
    s = torch.sigmoid(logits)                       # cumulative probs
    k_idx = torch.arange(logits.size(1),
                         device=logits.device).unsqueeze(0)

    lo = bounds[:, 0].unsqueeze(1)
    up = bounds[:, 1].unsqueeze(1)

    # targets: 1 for k ≤ lo, 0 for k > up, ignore inside (lo, up]
    mask_pos = (k_idx <= lo)          # target 1
    mask_neg = (k_idx >  up)          # target 0

    loss_pos = (F.binary_cross_entropy(s, torch.ones_like(s),
                                       reduction="none") * mask_pos)
    loss_neg = (F.binary_cross_entropy(s, torch.zeros_like(s),
                                       reduction="none") * mask_neg)

    denom = mask_pos.sum() + mask_neg.sum()
    return (loss_pos.sum() + loss_neg.sum()) / torch.clamp(denom, 1)



def _adjacent_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert cumulative logits → adjacent class probabilities."""
    s = torch.sigmoid(logits)
    s_pad = torch.cat([s, torch.zeros_like(s[:, :1])], dim=1)  # s_K = 0
    return s_pad[:, :-1] - s_pad[:, 1:]                       # shape [B, K]


def exp_rank_cum(logits: torch.Tensor, K_CLASSES: int | None = None) -> torch.Tensor:
    """
    Expected rank E[r] under the adjacent-class distribution.
    """
    p = _adjacent_probs(logits)
    if K_CLASSES is None:
        K_CLASSES = logits.size(1)
    idx = torch.arange(K_CLASSES, device=logits.device)
    return (p * idx).sum(1)


def ultra_tri_loss(la: torch.Tensor,
                   lh: torch.Tensor,
                   lr: torch.Tensor) -> torch.Tensor:
    """
    Enforces the general ultrametric condition by considering all three
    possible rootings/base edges for the triplet (la, lh, lr).
    The loss is minimized if any of the three ultrametric configurations are met.

    Internally, S_k = P(rank ≥ k) = σ(logit_k) is used.
    The condition aims for two S values to be equal and >= the third S value.
    (e.g. S_arm1 = S_arm2 >= S_base)

    Args:
        la (torch.Tensor): Cumulative logits for edge 'a' (e.g., q-r1). Shape [B, K]
        lh (torch.Tensor): Cumulative logits for edge 'h' (e.g., q-r2). Shape [B, K]
        lr (torch.Tensor): Cumulative logits for edge 'r' (e.g., r1-r2). Shape [B, K]

    Returns:
        torch.Tensor: Scalar mean loss over the batch.
    """
    Sa = torch.sigmoid(la)   # Cumulative probabilities for edge 'a'
    Sh = torch.sigmoid(lh)   # Cumulative probabilities for edge 'h'
    Sr = torch.sigmoid(lr)   # Cumulative probabilities for edge 'r'

    # --- Calculate loss for each of the 3 possible ultrametric configurations ---

    # Configuration 1: Sr is the "base" (Sa == Sh >= Sr)
    # This means S(r1,r2) is the smallest similarity (longest distance),
    # and S(q,r1) and S(q,r2) are equal and larger.
    eq_loss_sr_base = (Sa - Sh).pow(2)  # Sa should equal Sh
    # Sr should be less than or equal to Sa, and less than or equal to Sh
    ord_loss_sr_base = F.relu(Sr - Sa).pow(2) + F.relu(Sr - Sh).pow(2) 
    loss_config1 = (eq_loss_sr_base + ord_loss_sr_base).mean(dim=1) # Mean over K ranks for each sample

    # Configuration 2: Sh is the "base" (Sa == Sr >= Sh)
    # This means S(q,r2) is the smallest similarity,
    # and S(q,r1) and S(r1,r2) are equal and larger.
    eq_loss_sh_base = (Sa - Sr).pow(2)  # Sa should equal Sr
    # Sh should be less than or equal to Sa, and less than or equal to Sr
    ord_loss_sh_base = F.relu(Sh - Sa).pow(2) + F.relu(Sh - Sr).pow(2) 
    loss_config2 = (eq_loss_sh_base + ord_loss_sh_base).mean(dim=1)

    # Configuration 3: Sa is the "base" (Sh == Sr >= Sa)
    # This means S(q,r1) is the smallest similarity,
    # and S(q,r2) and S(r1,r2) are equal and larger.
    eq_loss_sa_base = (Sh - Sr).pow(2)  # Sh should equal Sr
    # Sa should be less than or equal to Sh, and less than or equal to Sr
    ord_loss_sa_base = F.relu(Sa - Sh).pow(2) + F.relu(Sa - Sr).pow(2) 
    loss_config3 = (eq_loss_sa_base + ord_loss_sa_base).mean(dim=1)
    
    # Stack the three possible loss configurations for each sample in the batch
    # Shape of losses_stacked: [3, BatchSize]
    losses_stacked = torch.stack([loss_config1, loss_config2, loss_config3], dim=0)
    
    # For each sample, find the minimum loss among the three configurations.
    # The model is penalized based on the "best fit" ultrametric interpretation.
    min_loss_per_sample, _ = torch.min(losses_stacked, dim=0)
    
    # Return the mean of these minimum losses over the batch
    return min_loss_per_sample.mean()


def CUM_TRE_LOSS(la: torch.Tensor,
                 lh: torch.Tensor,
                 lr: torch.Tensor,
                 b:  dict,
                 LAMBDA_INT: float = 1.0,
                 LAMBDA_TRI: float = 0.0) -> torch.Tensor:
    """
    • Interval-censored cumulative BCE on each triangle edge
    • Ultrametric triangle consistency via the distance-free `ultra_tri_loss`
    """
    ce = (cum_interval_bce(la, b["lqa"]) +
          cum_interval_bce(lh, b["lqh"]) +
          cum_interval_bce(lr, b["lrr"]))

    tri = ultra_tri_loss(la, lh, lr)

    return LAMBDA_INT * ce + LAMBDA_TRI * tri



