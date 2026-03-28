"""Full SE pipeline - PyTorch Lightning module.

Pipeline:  Audio → STFT → CFM VelocityNet → iSTFT → Audio
(No encoder/decoder — operates directly in STFT domain.)
"""

import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Settings
from src.models.aggregates.cfm import VelocityNet, direct_solve, ode_solve, sample_ot_interpolant
from src.models.utils.losses import (
    MultiResolutionSTFTLoss,
    cfm_loss,
    compressed_complex_loss,
    sigmoid_warmup,
)
from src.utils.audio_utils import (
    enhance_audio,
    istft_transform,
    stft_transform,
)


class SEModel(pl.LightningModule):
    """Conditional Flow Matching Speech Enhancement model (STFT-domain)."""

    def __init__(self, cfg: Settings) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg.model_dump())

        self.example_input_array = torch.randn(self.cfg.data.batch_size, self.cfg.data.segment_length)

        mc = cfg.model.velocity_net

        # ── VelocityNet (split-band, STFT domain) ──
        self.velocity_net = VelocityNet(
            hidden_dim=mc.hidden_dim,
            f_split=mc.f_split,
            time_emb_dim=mc.time_emb_dim,
            low_num_layers=mc.low_band.num_layers,
            low_num_heads=mc.low_band.num_heads,
            low_ff_mult=mc.low_band.ff_mult,
            low_dropout=mc.low_band.dropout,
            high_num_layers=mc.high_band.num_layers,
            high_dilations=mc.high_band.dilations,
            high_dropout=mc.high_band.dropout,
            bottleneck_num_heads=mc.bottleneck.num_heads,
            bottleneck_dropout=mc.bottleneck.dropout,
            decoder_num_layers=mc.decoder.num_layers,
            gradient_checkpointing=cfg.model.gradient_checkpointing,
        )

        # ── Losses ──
        stft_cfg = cfg.loss.multi_resolution_stft
        self.mr_stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=stft_cfg.fft_sizes,
            hop_sizes=stft_cfg.hop_sizes,
            win_sizes=stft_cfg.win_sizes,
        )

        # ── Audio buffer for AudioLoggingCallback (capped) ──
        self.val_audio_buffer: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            if getattr(m, "_is_film_output", False):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
                return
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ──────────────────────────────────────────────────────────────────────────
    # STFT helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _stft(self, wav: torch.Tensor) -> torch.Tensor:
        ac = self.cfg.audio
        return stft_transform(wav, ac.n_fft, ac.hop_length, ac.win_length, ac.window)

    def _istft(self, spec: torch.Tensor, length: int | None = None) -> torch.Tensor:
        ac = self.cfg.audio
        return istft_transform(spec, ac.n_fft, ac.hop_length, ac.win_length, ac.window, length=length)

    # ──────────────────────────────────────────────────────────────────────────
    # Velocity function builder (for ODE solvers)
    # ──────────────────────────────────────────────────────────────────────────

    def _velocity_fn(self, noisy_spec: torch.Tensor):
        """Return a callable (x_t, t) → v  for ODE solvers."""

        def fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.velocity_net(x_t, noisy_spec, t)

        return fn

    # ──────────────────────────────────────────────────────────────────────────
    # OT-CFM Training step
    # ──────────────────────────────────────────────────────────────────────────

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        clean_wav = batch["clean"]
        noisy_wav = batch["noisy"]

        # STFT → (B, 2, F, T)
        clean_spec = self._stft(clean_wav)
        noisy_spec = self._stft(noisy_wav)

        # OT-CFM interpolant + velocity prediction
        x_t, t, v_target, _ = sample_ot_interpolant(clean_spec, noisy_spec)
        v_pred = self.velocity_net(x_t, noisy_spec, t)  # (B, 2, F, T)

        epoch_float = self.current_epoch + (batch_idx / self.trainer.num_training_batches)
        lc = self.cfg.loss

        # ── CFM loss (always active) ──
        loss_cfm = cfm_loss(v_pred, v_target)
        w_cfm = sigmoid_warmup(epoch_float, lc.cfm.lambda_start, lc.cfm.lambda_end, lc.cfm.e0, lc.cfm.k)
        total_loss = w_cfm * loss_cfm

        # ── Curriculum: auxiliary losses only after cfm_only_epochs ──
        cfm_only = self.current_epoch < lc.curriculum_cfm_only_epochs

        if not cfm_only:
            # Correct x̂₁ projection: x̂₁ = x_t + (1 - t) · v_θ
            t_view = t[:, None, None, None]  # (B, 1, 1, 1)
            pred_spec = x_t + (1.0 - t_view) * v_pred  # (B, 2, F, T)

            # Per-sample temporal decay: t^y — losses near t→1 matter more
            gamma = lc.temporal_decay_gamma
            t_decay = t.pow(gamma)  # (B,)

            pred_wav = self._istft(pred_spec, length=clean_wav.shape[-1])
            min_wav_len = min(pred_wav.shape[-1], clean_wav.shape[-1])
            pred_wav = pred_wav[..., :min_wav_len]
            clean_wav_trimmed = clean_wav[..., :min_wav_len]

            min_t = min(pred_spec.shape[-1], clean_spec.shape[-1])
            pred_spec_trimmed = pred_spec[..., :min_t]
            clean_spec_trimmed = clean_spec[..., :min_t]

            # Multi-resolution STFT loss — (B,), weighted per sample by t_decay
            loss_stft = self.mr_stft_loss(pred_wav, clean_wav_trimmed)  # (B,)
            w_stft = sigmoid_warmup(
                epoch_float - lc.curriculum_cfm_only_epochs,
                lc.multi_resolution_stft.lambda_start,
                lc.multi_resolution_stft.lambda_end,
                lc.multi_resolution_stft.e0,
                lc.multi_resolution_stft.k,
            )
            total_loss = total_loss + w_stft * (t_decay * loss_stft).mean()

            # Compressed complex loss (force float32) — (B,), weighted per sample
            with torch.amp.autocast(device_type="cuda", enabled=False):
                loss_cc = compressed_complex_loss(
                    pred_spec_trimmed.float(),
                    clean_spec_trimmed.float(),
                    compression=lc.compressed_complex.compression,
                )
            w_cc = sigmoid_warmup(
                epoch_float - lc.curriculum_cfm_only_epochs,
                lc.compressed_complex.lambda_start,
                lc.compressed_complex.lambda_end,
                lc.compressed_complex.e0,
                lc.compressed_complex.k,
            )
            total_loss = total_loss + w_cc * (t_decay * loss_cc).mean()

        else:
            loss_stft = torch.tensor(0.0, device=total_loss.device)
            loss_cc = torch.tensor(0.0, device=total_loss.device)
            w_stft = torch.tensor(0.0, device=total_loss.device)
            w_cc = torch.tensor(0.0, device=total_loss.device)

        # ── Logging ──
        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/cfm_loss", loss_cfm, prog_bar=True)
        self.log("train/stft_loss", loss_stft.mean(), prog_bar=True)
        self.log("train/cc_loss", loss_cc.mean(), prog_bar=True)
        self.log("train/w_cfm", w_cfm, prog_bar=False)
        self.log("train/w_stft", w_stft, prog_bar=False)
        self.log("train/w_cc", w_cc, prog_bar=False)


        return total_loss

    def on_train_epoch_end(self):
        if not self.logger:
            return
        if self.current_epoch % 10 != 0:
            return
        for name, params in self.named_parameters():
            if params.requires_grad:
                self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def on_before_optimizer_step(self, optimizer) -> None:
        if self.global_step % 1000 != 0:
            return
        if not self.logger:
            return
        for name, params in self.named_parameters():
            if params.requires_grad and params.grad is not None:
                self.logger.experiment.add_histogram(f"{name}_grad", params.grad, self.global_step)
                self.logger.experiment.add_histogram(f"{name}", params, self.global_step)

    def on_train_start(self) -> None:
        if self.logger and hasattr(self.logger, "log_dir") and self.logger.log_dir:
            run_dir = self.logger.log_dir
            os.makedirs(run_dir, exist_ok=True)

            if self.cfg.model.save_onnx:
                onnx_path = os.path.join(run_dir, "model_architecture.onnx")
                self.eval()
                try:
                    dummy_input = self.example_input_array.to(self.device)
                    torch.onnx.export(
                        self,
                        (dummy_input,),
                        onnx_path,
                        opset_version=18,
                        input_names=["noisy_audio"],
                        output_names=["clean_audio"],
                    )
                    self.print(f"ONNX exported to: {onnx_path}")
                except Exception as e:
                    self.print(f"ONNX export error: {e}")
                self.train()

        return super().on_train_start()

    # ──────────────────────────────────────────────────────────────────────────
    # Inference (ODE integration)
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def enhance(self, noisy_wav: torch.Tensor) -> torch.Tensor:
        """Enhance a short waveform (B, T) — fits in memory (validation path)."""
        noisy_spec = self._stft(noisy_wav)  # (B, 2, F, T)

        vfn = self._velocity_fn(noisy_spec)
        ode_val = self.cfg.ode.val
        pred_spec = ode_solve(
            vfn,
            noisy_spec,
            method=ode_val.solver,
            num_steps=ode_val.num_steps,
            atol=ode_val.atol,
            rtol=ode_val.rtol,
        )  # (B, 2, F, T)

        return self._istft(pred_spec, length=noisy_wav.shape[-1])

    @torch.no_grad()
    def enhance_fast(self, noisy_wav: torch.Tensor) -> torch.Tensor:
        """Single-step enhancement via direct_solve (one network call).

        Faster than enhance() but lower quality. Useful for quick previews
        or as a warm-start before more expensive ODE integration.
        """
        noisy_spec = self._stft(noisy_wav)  # (B, 2, F, T)
        vfn = self._velocity_fn(noisy_spec)
        pred_spec = direct_solve(vfn, noisy_spec)
        return self._istft(pred_spec, length=noisy_wav.shape[-1])

    @torch.no_grad()
    def enhance_long(
        self,
        noisy_wav: torch.Tensor,
        solver: str | None = None,
        num_steps: int | None = None,
        chunk_size: int | None = None,
        overlap: float | None = None,
        max_batch_chunks: int | None = None,
        atol: float | None = None,
        rtol: float | None = None,
    ) -> torch.Tensor:
        """Enhance variable-length audio using Overlap-Add (OLA)."""
        
        if noisy_wav.ndim == 2:
            noisy_wav = noisy_wav.squeeze(0)

        ode_test = self.cfg.ode.test
        inf_cfg = self.cfg.inference

        # 1. Resolve os parâmetros finais (Argumento vs Config)
        final_chunk_size = chunk_size if chunk_size is not None else inf_cfg.chunk_size
        final_overlap = overlap if overlap is not None else inf_cfg.overlap

        # 2. Calcula o hop_size garantindo que seja um INTEIRO
        hop_size = int(final_chunk_size * (1.0 - final_overlap))

        # 3. Faz o padding para remover possíveis estalos de contorno
        original_length = noisy_wav.shape[-1]
        noisy_wav_padded = F.pad(noisy_wav, (hop_size, hop_size), mode='constant', value=0.0)

        # 4. Processa o áudio (A mágica do WOLA acontece aqui dentro)
        resulting = enhance_audio(
            noisy_wav=noisy_wav_padded,
            model=self,
            chunk_size=final_chunk_size,
            overlap=final_overlap,
            solver=solver or ode_test.solver,
            num_steps=num_steps or ode_test.num_steps,
            max_batch_chunks=max_batch_chunks or inf_cfg.max_batch_chunks,
            atol=atol if atol is not None else ode_test.atol,
            rtol=rtol if rtol is not None else ode_test.rtol,
        )

        # 5. Remove padding para limpar o áudio reconstruído
        return resulting[..., hop_size : hop_size + original_length]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Real inference path: waveform in → enhanced waveform out.

        This is the canonical forward pass used by torch.compile and
        standard PyTorch inference. For variable-length audio, use
        enhance_long() which handles chunked OLA processing.
        """
        return self.enhance(x)

    # ──────────────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────────────

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        clean_wav = batch["clean"]
        noisy_wav = batch["noisy"]

        # ----------------------------------------------------------------------
        # 1. VIA RÁPIDA: CALCULAR A LOSS DO CFM (O que o Checkpoint vai monitorar)
        # ----------------------------------------------------------------------
        clean_spec = self._stft(clean_wav)
        noisy_spec = self._stft(noisy_wav)

        # Amostra um tempo t aleatório e cria o alvo (igual no training_step)
        x_t, t, v_target, _ = sample_ot_interpolant(clean_spec, noisy_spec)
        
        # Faz apenas UMA passagem pela rede
        v_pred = self.velocity_net(x_t, noisy_spec, t)

        # Calcula a loss principal
        val_cfm_loss = torch.nn.functional.mse_loss(v_pred, v_target)
        self.log("val_loss/cfm", val_cfm_loss, prog_bar=True, sync_dist=True, batch_size=noisy_wav.size(0))

        # (Opcional) Calcular as losses auxiliares puras para o TensorBoard
        # Usamos a projeção de 1-passo para ter uma noção da qualidade sem rodar a EDO
        t_view = t.view(-1, 1, 1, 1)
        pred_spec_1step = x_t + (1.0 - t_view) * v_pred
        pred_wav_1step = self._istft(pred_spec_1step, length=clean_wav.shape[-1])
        
        val_mr_stft = self.mr_stft_loss(pred_wav_1step, clean_wav).mean()
        val_cc = compressed_complex_loss(pred_spec_1step, clean_spec).mean()
        
        self.log("val_loss/mr_stft", val_mr_stft, sync_dist=True, batch_size=noisy_wav.size(0))
        self.log("val_loss/cc", val_cc, sync_dist=True, batch_size=noisy_wav.size(0))

        # ----------------------------------------------------------------------
        # 2. VIA LENTA: GERAR ÁUDIO COMPLETO APENAS PARA O BUFFER
        # ----------------------------------------------------------------------
        max_buf = self.cfg.validation.audio_log_samples
        
        # O pulo do gato: Só chama o solver da EDO se o buffer ainda não estiver cheio!
        if len(self.val_audio_buffer) < max_buf:
            # Aqui sim, roda os 20 passos do ODE solver
            pred_wav_ode = self.enhance(noisy_wav)

            min_len = min(pred_wav_ode.shape[-1], clean_wav.shape[-1])
            self.val_audio_buffer.append(
                (
                    clean_wav[0, :min_len].detach().cpu(),
                    noisy_wav[0, :min_len].detach().cpu(),
                    pred_wav_ode[0, :min_len].detach().cpu(),
                )
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Optimiser / Scheduler
    # ──────────────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        tc = self.cfg.training
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=tc.learning_rate,
            weight_decay=tc.weight_decay,
        )

        if tc.lr_scheduler == "cosine_with_warmup":
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                total_iters=tc.warmup_steps,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=tc.max_epochs,
                eta_min=tc.lr_min,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[tc.warmup_steps],
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        elif tc.lr_scheduler == "cosine_annealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=tc.max_epochs,
                eta_min=tc.lr_min,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=tc.lr_factor,
                patience=tc.lr_patience,
                min_lr=tc.lr_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": tc.early_stopping_metric,
                    "interval": "epoch",
                },
            }
