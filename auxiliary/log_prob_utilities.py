# -----------------------------------------------------------------------------
# PyTorch Imports
# -----------------------------------------------------------------------------
import torch              # # tensors, device control
import torch.nn.functional as F  # # (used later) one-hot, etc.

# -----------------------------------------------------------------------------
# Numpy Imports
# -----------------------------------------------------------------------------
import numpy as np

# =============================================================================
# Log-prob utilities (categorical / gaussian) with device-safe constants
# =============================================================================

# -----------------------------------------------------------------------------
# Numerical stability constant
# -----------------------------------------------------------------------------
EPS = 1e-5

def log_categorical_indices(x_idx, logits):
    # x_idx: (B, D) long in {0..V-1}
    # logits: (B, D, V)
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, x_idx.unsqueeze(-1)).squeeze(-1).sum(-1)  # (B,) 


def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    # -------------------------------------------------------------------------
    # Diagonal Gaussian log-density (elementwise):
    #
    # Inputs:
    # - x, mu, log_var: (B, D)
    #
    # Output:
    # - log_p: (B, D) elementwise
    #
    # Note:
    # - constants created on x.device to avoid device mismatch errors.
    # -------------------------------------------------------------------------

    # # create scalar constant log(2*pi) on correct device/dtype
    log2pi = torch.log(x.new_tensor(2.0 * np.pi))
    
    log_p = -0.5 * (log2pi + log_var + torch.exp(-log_var) * (x - mu) ** 2)

    if reduction == "avg":
        return torch.mean(log_p, dim=dim)
    elif reduction == "sum":
        return torch.sum(log_p, dim=dim)
    else:
        return log_p


def log_standard_normal(x, reduction=None, dim=None):
    # -------------------------------------------------------------------------
    # Standard Normal log-density (elementwise):
    #
    # Input:
    # - x: (B, D)
    #
    # Output:
    # - log_p: (B, D)
    # -------------------------------------------------------------------------

    
    log2pi = torch.log(x.new_tensor(2.0 * np.pi))

    log_p = -0.5 * (log2pi + x**2)


    if reduction == "avg":
        return torch.mean(log_p, dim=dim)
    elif reduction == "sum":
        return torch.sum(log_p, dim=dim)
    else:
        return log_p