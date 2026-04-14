import torch


@torch.no_grad()
def ode_solve(model: torch.nn.Module, noisy_stft: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
    """
    Resolve a EDO usando o método de Euler para gerar o áudio limpo.
    """
    B = noisy_stft.shape[0]
    device = noisy_stft.device

    # Em t=0, o nosso estado inicial é o áudio ruidoso
    x_t = noisy_stft.clone()

    # CORREÇÃO 1: N passos precisam de N+1 pontos no tempo!
    t_span = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    dt = t_span[1] - t_span[0]  # Agora não dá mais erro de índice com n_steps=1

    # CORREÇÃO 2: O loop roda exatamente n_steps vezes
    for i in range(n_steps):
        # O tempo atual no formato que o SinusoidalEmbedding espera: (B,)
        t_current = t_span[i].expand(B)

        # A U-Net SEMPRE espera a condição concatenada com o estado atual
        model_input = torch.cat([noisy_stft, x_t], dim=1)

        # Predizemos o campo vetorial (a direção para o áudio limpo)
        v_pred = model(model_input, t_current)

        # Passo de Euler: x_{t+1} = x_t + v * dt
        x_t = x_t + v_pred * dt

    # Quando o loop termina, x_t chegou em t=1 (O áudio aprimorado!)
    return x_t