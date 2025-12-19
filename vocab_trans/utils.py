import torch

def pmi_similarity(X: torch.Tensor,
                   eps: float = 1e-12,
                   positive_only: bool = True) -> torch.Tensor:
    """
    P(i,j) = X_ij / sum_ab X_ab
    P(i)   = sum_j P(i,j)
    P(j)   = sum_i P(i,j)
    PMI(i,j) = log( P(i,j) / (P(i)*P(j)) )
    """
    X = (X - X.min()) / (X.max() - X.min())
    X = X.float()
    total = X.sum()
    if total <= 0:
        return torch.zeros_like(X)

    P = X / total                                  # [n, n]
    Pi = P.sum(dim=1, keepdim=True)               # [n, 1]
    Pj = P.sum(dim=0, keepdim=True)               # [1, n]

    denom = Pi * Pj                                # [n, n]
    denom = torch.clamp(denom, min=eps)

    ratio = P / denom
    ratio = torch.clamp(ratio, min=eps)

    PMI = torch.log(ratio)

    if positive_only:
        PMI = torch.clamp(PMI, min=0.0)           # PPMI

    return PMI


def zscore_column_similarity(X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    μ_j  = mean_i X_ij
    σ_j  = std_i  X_ij
    Z_ij = (X_ij - μ_j) / (σ_j + eps)
    """
    X = X.float()
    mean = X.mean(dim=0, keepdim=True)                    # [1, n]
    std = X.std(dim=0, unbiased=False, keepdim=True)      # [1, n]
    std = std + eps
    Z = (X - mean) / std
    return Z / (Z.max(dim=-1).values.view(-1, 1))