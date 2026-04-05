import torch
import torch.nn.functional as F


def cfm_loss(v_pred: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
    """
    Calcula a perda do Conditional Flow Matching.

    Args:
        v_pred: O campo vetorial predito pela U-Net (B, C, F, T)
        x_0: Condição inicial (Áudio ruidoso) (B, C, F, T)
        x_1: Destino (Áudio limpo) (B, C, F, T)
    """
    # O alvo real é a derivada da trajetória linear no tempo
    v_target = x_1 - x_0

    # MSE Loss é o padrão para treinar o campo vetorial
    loss = F.mse_loss(v_pred, v_target)

    return loss
