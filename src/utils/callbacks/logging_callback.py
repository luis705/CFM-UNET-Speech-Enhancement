"""Audio logging callback — logs waveforms, spectrograms, and fast metrics to TensorBoard."""

import gc
import io
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pytorch_lightning as pl
import torch
import torchaudio
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from src.data.transforms import signal_to_mel


class LoggingCallback(pl.Callback):
    """Log audio samples and metrics at end of test and val dataset.

    Reads ``pl_module.val_audio_buffer`` — a list of ``(clean, noisy, pred)`` CPU tensors
    that ``validation_step`` populates (capped at ``audio_log_samples``).
    """

    def __init__(self, n_samples: int = 4, histogram_logging_frequency: int = 0, config_path: str = "settings.yaml"):
        super().__init__()
        self.n_samples = n_samples
        self.histogram_logging_frequency = histogram_logging_frequency
        self.config_path = config_path
        self._mel_trans = None
        self._amp_to_db = None

    def on_train_start(self, trainer, pl_module):
        tb = pl_module.logger.experiment if pl_module.logger else None

        if tb is None:
            return

        log_dir = trainer.logger.log_dir  # dir do run atual
        if os.path.exists(self.config_path):
            shutil.copy(self.config_path, os.path.join(log_dir, "config.yaml"))
            # Opcional: logar conteúdo como texto também
            with open(self.config_path) as f:
                config_content = f.read()
            tb.add_text("config", config_content, global_step=0)

        tb.add_text("model_summary", str(pl_module), global_step=0)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.histogram_logging_frequency > 0 and batch_idx % self.histogram_logging_frequency == 0:
            tb = pl_module.logger.experiment if pl_module.logger else None

            if tb is None:
                return

            model = pl_module.model
            hists = {
                "stem": model.stem[0].weight,
                "output_projection": model.output_projection[-1].weight,
                "bottleneck_conv": model.bottleneck.conv1[0].weight,
                "bottleneck_attn": model.bottleneck.attn.attn_time.qkv.weight,
                "gate": model.decoders[0].gate.w_gate.weight,
                "film": model.encoders[0].film.out.weight,
            }

            step = pl_module.global_step

            for name, param in hists.items():
                # 1. Log dos Pesos (Histograma e Norma L2)
                tb.add_histogram(f"weights/{name}", param.detach().cpu(), global_step=step)

                weight_norm = torch.linalg.vector_norm(param).item()
                tb.add_scalar(f"norms_weight/{name}", weight_norm, global_step=step)

                # 2. Log dos Gradientes (Histograma e Norma L2)
                if param.grad is not None:
                    tb.add_histogram(f"grads/{name}", param.grad.detach().cpu(), global_step=step)

                    grad_norm = torch.linalg.vector_norm(param.grad).item()
                    tb.add_scalar(f"norms_grad/{name}", grad_norm, global_step=step)

            tb.flush()

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Garante que o buffer de áudio comece vazio em cada validação."""
        buf = getattr(pl_module, "val_audio_buffer", None)
        if buf is not None:
            buf.clear()

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Garante que o buffer de áudio comece vazio em cada teste."""
        buf = getattr(pl_module, "test_audio_buffer", None)
        if buf is not None:
            buf.clear()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.save_from_buffer(pl_module, "val_audio_buffer", "val_binaries")

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Processa o fim do teste. Salva localmente APENAS se não houver logger ativo."""
        local_path = None

        # Se o trainer não tem logger (script test.py), definimos o caminho para salvar arquivos
        if trainer.logger is None:
            # Busca o diretório definido em settings.yaml (test.save_dir)
            try:
                local_path = pl_module.cfg.test.save_dir
            except (AttributeError, KeyError):
                local_path = "test_results/samples"
            
            os.makedirs(local_path, exist_ok=True)
            print(f"\n[Logging] Salvando resultados de teste localmente em: {local_path}")

        # 1. Salva amostras de áudio
        # Se local_path for None (caso do train.py), save_from_buffer usará apenas o TB
        self.save_from_buffer(pl_module, "test_audio_buffer", "test_binaries", local_path=local_path)

        # 2. Salva métricas (CSV/Histogramas)
        self.save_test_metrics(pl_module, local_path=local_path)

    # ---------- AUX METHODS ----------
    def save_from_buffer(self, pl_module: pl.LightningModule, buf_name: str, tb_preffix: str, local_path: str = None) -> None:
        buf = getattr(pl_module, buf_name, [])
        if not buf:
            return

        tb = pl_module.logger.experiment if pl_module.logger and hasattr(pl_module.logger, "experiment") else None

        sr = pl_module.audio_sample_rate
        step = pl_module.global_step

        samples_to_log = buf[: self.n_samples]

        for i, buf_data in enumerate(samples_to_log):
            suffix = f"sample_{i}"
            metrics = buf_data.get("metrics", {})
            
            clean, noisy, enhanced = buf_data["clean"], buf_data["noisy"], buf_data["enhanced"]
            
            # Se o módulo estiver operando em modo Mel, os tensores no buffer são espectrogramas
            if getattr(pl_module, "input_mel", False):
                with torch.no_grad():
                    mel_cfg = pl_module.cfg.audio.mel.model_dump()
                    def to_vocoded(wav):
                        # Converte waveform para mel (formato esperado pelo signal_to_mel)
                        mel_params = {k: v for k, v in mel_cfg.items() if k != "vocoder_model_path"}
                        m = signal_to_mel(
                            wav.unsqueeze(0).to(pl_module.device), 
                            pl_module.audio_sample_rate, 
                            n_fft=pl_module.fft_size,
                            hop_length=pl_module.hop_length,
                            win_length=pl_module.window_size,
                            **mel_params
                        ) # Resulta em (1, 1, Mels, T)
                        # Transpõe para (1, T, Mels) e vocoda
                        return pl_module.vocoder(m.squeeze(1).transpose(1, 2)).waveforms.squeeze(0).cpu()

                    clean = to_vocoded(clean)
                    noisy = to_vocoded(noisy)
                    enhanced = to_vocoded(enhanced)

            if tb is not None:
                for group_metrics in metrics.values():
                    for m_name, m_val in group_metrics.items():
                        tb.add_scalar(f"{tb_preffix}_{suffix}/{m_name}", m_val, step)

            self._log_rich_audio_data(
                tb,
                clean,
                noisy,
                enhanced,
                step,
                sr,
                suffix=suffix,
                tb_preffix=tb_preffix,
                local_path=local_path,
            )

        buf.clear()
        plt.close('all')
        gc.collect()

    def _log_rich_audio_data(
        self,
        tb,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        enhanced: torch.Tensor,
        step: int,
        sr: int,
        suffix: str,
        tb_preffix: str,
        local_path: str = None,
    ) -> None:
        # torchaudio.save espera tensores 1D ou 2D. Se vier (1, 1, T), removemos o batch.
        if clean.ndim == 3: clean = clean.squeeze(0)
        if noisy.ndim == 3: noisy = noisy.squeeze(0)
        if enhanced.ndim == 3: enhanced = enhanced.squeeze(0)

        def plot_waveform(waveform, title, save_path=None):
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.plot(waveform.flatten().cpu().numpy(), linewidth=0.5)
            ax.set_title(title)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlim(0, waveform.shape[-1])
            ax.set_ylim(-1, 1)
            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            with Image.open(buf) as img:
                return ToTensor()(img)

        if local_path:
            sample_dir = os.path.join(local_path, suffix)
            os.makedirs(sample_dir, exist_ok=True)
            torchaudio.save(os.path.join(sample_dir, "clean.wav"), clean.cpu(), sr)
            torchaudio.save(os.path.join(sample_dir, "noisy.wav"), noisy.cpu(), sr)
            torchaudio.save(os.path.join(sample_dir, "enhanced.wav"), enhanced.cpu(), sr)

        # --- Audio & Waveform Plots (Logger) ---
        if tb is not None:
            tb.add_audio(f"{tb_preffix}_{suffix}/audio_clean", clean.flatten().unsqueeze(0), step, sample_rate=sr)
            tb.add_audio(f"{tb_preffix}_{suffix}/audio_noisy", noisy.flatten().unsqueeze(0), step, sample_rate=sr)
            tb.add_audio(f"{tb_preffix}_{suffix}/audio_enhanced", enhanced.flatten().unsqueeze(0), step, sample_rate=sr)
            tb.add_audio(f"{tb_preffix}_{suffix}/audio_enhanced_noisy", (enhanced - noisy).flatten().unsqueeze(0), step, sample_rate=sr)
            tb.add_audio(f"{tb_preffix}_{suffix}/audio_enhanced_clean", (enhanced - clean).flatten().unsqueeze(0), step, sample_rate=sr)
            tb.add_audio(f"{tb_preffix}_{suffix}/audio_clean_noisy", (clean - noisy).flatten().unsqueeze(0), step, sample_rate=sr)
            
            tb.add_image(f"{tb_preffix}_{suffix}/wave_clean", plot_waveform(clean, "Clean"), step)
            tb.add_image(f"{tb_preffix}_{suffix}/wave_noisy", plot_waveform(noisy, "Noisy"), step)
            tb.add_image(f"{tb_preffix}_{suffix}/wave_enhanced", plot_waveform(enhanced, "Enhanced"), step)
            tb.add_image(f"{tb_preffix}_{suffix}/wave_enhanced_noisy", plot_waveform(enhanced - noisy, "Enhanced - Noisy"), step)
            tb.add_image(f"{tb_preffix}_{suffix}/wave_enhanced_clean", plot_waveform(enhanced - clean, "Enhanced - Clean"), step)
            tb.add_image(f"{tb_preffix}_{suffix}/wave_clean_noisy", plot_waveform(clean - noisy, "Clean - Noisy"), step)

        # --- Waveform Plots (Local) ---
        if local_path:
            plot_waveform(clean, "Clean", os.path.join(sample_dir, "wave_clean.png"))
            plot_waveform(noisy, "Noisy", os.path.join(sample_dir, "wave_noisy.png"))
            plot_waveform(enhanced, "Enhanced", os.path.join(sample_dir, "wave_enhanced.png"))

        # --- Spectrograms ---
        if self._mel_trans is None:
            self._mel_trans = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=512, hop_length=128, n_mels=80)
        if self._amp_to_db is None:
            self._amp_to_db = torchaudio.transforms.AmplitudeToDB()

        device = clean.device
        spec_clean = self._amp_to_db.to(device)(self._mel_trans.to(device)(clean.flatten().unsqueeze(0)))
        spec_enhanced = self._amp_to_db.to(device)(self._mel_trans.to(device)(enhanced.flatten().unsqueeze(0)))
        spec_noisy = self._amp_to_db.to(device)(self._mel_trans.to(device)(noisy.flatten().unsqueeze(0)))
        spec_enhanced_noisy = self._amp_to_db.to(device)(self._mel_trans.to(device)((enhanced - noisy).flatten().unsqueeze(0)))
        spec_enhanced_clean = self._amp_to_db.to(device)(self._mel_trans.to(device)((enhanced - clean).flatten().unsqueeze(0)))
        spec_clean_noisy = self._amp_to_db.to(device)(self._mel_trans.to(device)((clean - noisy).flatten().unsqueeze(0)))


        # Log espectrogramas normais
        for tag, spec in [
            ("clean", spec_clean),
            ("noisy", spec_noisy),
            ("enhanced", spec_enhanced),
            ("enhanced_noisy", spec_enhanced_noisy),
            ("enhanced_clean", spec_enhanced_clean),
            ("clean_noisy", spec_clean_noisy),
        ]:
            # Escala dB fixa para comparação consistente (-80 a 0 dB)
            v_min, v_max = -80.0, 0.0
            img = (spec - v_min) / (v_max - v_min + 1e-8)
            img = torch.clamp(img, 0, 1)

            if tb is not None:
                tb.add_image(f"{tb_preffix}_{suffix}/spec_{tag}", img, step)
            if local_path:
                save_image(img, os.path.join(sample_dir, f"spec_{tag}.png"))

    def save_test_metrics(self, pl_module: pl.LightningModule, local_path: str = None) -> None:
        tb = pl_module.logger.experiment if pl_module.logger and hasattr(pl_module.logger, "experiment") else None

        # 1. Coleta métricas do pl_module.test_metrics_buffer
        metrics_list = getattr(pl_module, "test_metrics_buffer", [])
        if not metrics_list:
            return

        # Agrupa os valores de cada métrica (achatando o dicionário aninhado)
        all_metrics = {}
        for m_dict in metrics_list:
            for group_name, group_vals in m_dict.items():
                for metric_name, val in group_vals.items():
                    full_name = f"{group_name}/{metric_name}"
                    if full_name not in all_metrics:
                        all_metrics[full_name] = []
                    
                    val_scalar = val.item() if torch.is_tensor(val) else val
                    all_metrics[full_name].append(val_scalar)

        # 2. Gera Histogramas iterando sobre as métricas coletadas
        for name, values in all_metrics.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(values, bins=min(len(values), 30), color="skyblue", edgecolor="black")
            ax.set_title(f"Distribution of {name} (Mean: {np.mean(values):.4f})")
            ax.set_xlabel(name)
            ax.set_ylabel("Frequency")

            safe_name = name.replace("/", "_")
            if tb:
                tb.add_figure(f"test/hist_{safe_name}", fig, global_step=pl_module.global_step)
            if local_path:
                fig.savefig(os.path.join(local_path, f"hist_{safe_name}.png"))
            plt.close(fig)

        # 4. Salva o CSV como texto no TensorBoard
        df = pd.DataFrame(all_metrics)
        csv_text = df.to_csv(index=False)
        if tb:
            tb.add_text("test/individual_metrics", f"```csv\n{csv_text}\n```", global_step=pl_module.global_step)
        
        if local_path:
            df.to_csv(os.path.join(local_path, "test_metrics.csv"), index=False)

        # 5. Limpa buffer para liberar memória
        pl_module.test_metrics_buffer.clear()
