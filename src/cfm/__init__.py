"""Inicialização do pacote CFM.

Aqui aplicamos configurações globais do pacote, como supressão de
warnings conhecidos do PyTorch que não afetam o comportamento do modelo.
"""
from __future__ import annotations

import warnings

# Suprime aviso específico do PyTorch sobre tipos mistos em masks de atenção:
# "Support for mismatched key_padding_mask and attn_mask is deprecated. Use same type for both instead."
# Filtramos por mensagem para evitar silenciar outros UserWarnings.
warnings.filterwarnings(
	"ignore",
	category=UserWarning,
	message=r".*key_padding_mask and attn_mask.*",
)

__all__ = []
