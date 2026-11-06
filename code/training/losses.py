"""Loss functions and a single optimisation step shared by warm start and self-play training."""
import numpy as np
import torch
import torch.nn.functional as F


def dense_policy_target(pol_idx: np.ndarray, pol_p: np.ndarray, n_moves: int) -> torch.Tensor:
    target = torch.zeros(len(pol_idx), n_moves)
    idx = torch.from_numpy(pol_idx.astype(np.int64))
    p = torch.from_numpy(pol_p)
    valid = idx >= 0
    target.scatter_(1, idx.clamp(min=0), p * valid)
    return target


def train_step(net, optimizer, planes, policy_target, wdl, device, value_weight=1.0):
    """policy_target: dense distribution (B, n_moves) or move indices (B,)."""
    net.train()
    x = torch.as_tensor(planes, device=device)
    logits, wdl_logits = net(x)
    logp = F.log_softmax(logits, dim=-1)
    pt = torch.as_tensor(policy_target, device=device)
    if pt.dim() == 1:
        policy_loss = F.nll_loss(logp, pt.long())
    else:
        policy_loss = -(pt * logp).sum(-1).mean()
    value_loss = F.cross_entropy(wdl_logits, torch.as_tensor(wdl, device=device).long())
    loss = policy_loss + value_weight * value_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), float("inf"))  # measure only
    optimizer.step()
    with torch.no_grad():
        target = pt.long() if pt.dim() == 1 else pt.argmax(-1)
        policy_acc = (logits.argmax(-1) == target).float().mean()
        value_acc = (wdl_logits.argmax(-1) == torch.as_tensor(wdl, device=device).long()).float().mean()
    return {"loss": loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
            "grad_norm": grad_norm.item(), "policy_acc": policy_acc.item(), "value_acc": value_acc.item()}
