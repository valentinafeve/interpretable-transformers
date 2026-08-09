import torch


def effective_rank(A: torch.Tensor) -> torch.Tensor:
    """
    Effective rank de Roy & Vetterli (2007): erank(A) = exp(H(p)),
    donde p_k = sigma_k / ||sigma||_1 son los valores singulares de A
    normalizados y H es su entropía de Shannon (con la convención 0*log(0) = 0).

    Soporta batches: A puede tener shape (..., M, N), y el rango efectivo
    se calcula sobre las últimas dos dimensiones, devolviendo shape (...,).
    Cumple 1 <= erank(A) <= rank(A) <= min(M, N).
    """
    singular_values = torch.linalg.svdvals(A)  # (..., Q)
    p = singular_values / singular_values.sum(dim=-1, keepdim=True)

    log_p = torch.where(p > 0, torch.log(p), torch.zeros_like(p))
    entropy = -(p * log_p).sum(dim=-1)

    return torch.exp(entropy)
