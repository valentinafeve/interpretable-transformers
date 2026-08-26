import torch


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Centered Kernel Alignment (Kornblith et al., 2019) con kernel lineal.

    X: (..., n, p1), Y: (..., n, p2) -- n ejemplos, p1/p2 features por representación.
    Se centran las columnas (media 0 por feature) antes de calcular la similitud.

    CKA(XX^T, YY^T) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    Devuelve un escalar en [0, 1]: 1 si las representaciones son idénticas
    salvo rotación/escala isotrópica, cerca de 0 si son independientes.
    Soporta batch (p. ej. X: (H, 1, n, p), Y: (1, H, n, p) -> (H, H) por
    broadcasting, para obtener la matriz de similitud entre pares de cabezas).
    """
    X = X - X.mean(dim=-2, keepdim=True)
    Y = Y - Y.mean(dim=-2, keepdim=True)

    cross_term = torch.linalg.matrix_norm(Y.transpose(-2, -1) @ X) ** 2
    xtx_norm = torch.linalg.matrix_norm(X.transpose(-2, -1) @ X)
    yty_norm = torch.linalg.matrix_norm(Y.transpose(-2, -1) @ Y)

    return cross_term / (xtx_norm * yty_norm)
