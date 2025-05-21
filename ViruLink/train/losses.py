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
    Enforces the rooted ultrametric condition

        S(q,r1) == S(q,r2) >= S(r1,r2)

    where S_k = P(rank ≥ k) = σ(logit_k).
    """
    Sa = torch.sigmoid(la)   # cumulative probs: (q , r1)
    Sh = torch.sigmoid(lh)   # cumulative probs: (q , r2)
    Sr = torch.sigmoid(lr)   # cumulative probs: (r1, r2)

    # equality term
    eq_loss  = (Sa - Sh).pow(2).mean(1)

    # ordering term  (sign fixed)
    ord_loss = (F.relu(Sr - Sa).pow(2).mean(1) +   # Sr should not exceed Sa
                F.relu(Sr - Sh).pow(2).mean(1))    # Sr should not exceed Sh

    return (eq_loss + ord_loss).mean()

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



