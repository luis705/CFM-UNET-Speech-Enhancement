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
import torch.nn.functional as F
import torch
import torchaudio
from torchvision.transforms import ToTensor


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
        self.mel_trans = None
        self._static_cache = {}

    def _get_mel_trans(self, cfg):
        if self.mel_trans is None:
            self.mel_trans = torchaudio.transforms.MelSpectrogram(
                sample_rate=cfg.sample_rate,
                n_fft=cfg.fft_size,
                hop_length=cfg.hop_length,
                n_mels=80,
            )
        return self.mel_trans

    def _plot_waveform_to_tensor(self, waveform, title, save_path=None):
        if waveform.ndim == 2: waveform = waveform.squeeze(0)
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.plot(waveform.flatten().cpu().float().numpy(), linewidth=0.5)
        ax.set_title(title)
        ax.set_ylim(-1, 1)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        with Image.open(buf) as img:
            return ToTensor()(img)

    def _stft_to_images(self, waveform, n_fft: int = 1024, hop_length: int = 120):
        waveform = waveform.flatten().to(torch.float32)
        stft = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True,
            window=torch.hann_window(n_fft, device=waveform.device),
        )

        mag = torch.abs(stft)
        db_mag = 10.0 * torch.log10(torch.clamp(mag, min=1e-5))
        v_max = db_mag.max()
        v_min = v_max - 80.0
        mag_img = torch.clamp((db_mag - v_min) / 80.0, 0, 1)
        mag_img = torch.flip(mag_img, dims=[0]).unsqueeze(0)

        phase = torch.angle(stft)
        phase_img = (phase + torch.pi) / (2 * torch.pi)
        phase_img = torch.flip(phase_img, dims=[0]).unsqueeze(0)

        return mag_img, phase_img

    def _logger_base_dir(self, trainer: pl.Trainer, fallback_dir: str = "tb_logs") -> str:
        logger = getattr(trainer, "logger", None)
        log_dir = getattr(logger, "log_dir", None)
        if log_dir:
            return log_dir
        return fallback_dir

    def _tensorboard_backend(self, pl_module: pl.LightningModule):
        logger = getattr(pl_module, "logger", None)
        experiment = getattr(logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "add_scalar"):
            return experiment
        return None

    def _mlflow_logger(self, pl_module: pl.LightningModule):
        logger = getattr(pl_module, "logger", None)
        if logger is None:
            return None
        if hasattr(logger, "log_metrics") and self._tensorboard_backend(pl_module) is None:
            return logger
        return None

    def _log_mlflow_artifact(self, pl_module: pl.LightningModule, file_path: str, artifact_path: str) -> None:
        logger = self._mlflow_logger(pl_module)
        if logger is None or not os.path.exists(file_path):
            return
        import mlflow

        run_id = getattr(logger, "run_id", None)
        if run_id is None:
            return
        try:
            active_run = mlflow.active_run()
            if active_run is None:
                active_run = mlflow.start_run(run_id=run_id)

            if active_run and active_run.info.status == "FAILED":
                client = mlflow.tracking.MlflowClient()
                client.update_run(active_run.info.run_id, status="RUNNING")

            mlflow.log_artifact(file_path, artifact_path=artifact_path)
        except Exception:
            pass

    def _log_mlflow_image(self, pl_module: pl.LightningModule, image, name: str, step: int | None = None) -> None:
        logger = self._mlflow_logger(pl_module)
        if logger is None:
            return
        try:
            import mlflow
            run_id = getattr(logger, "run_id", None)
            if run_id is None:
                return

            active_run = mlflow.active_run()
            if active_run is None:
                active_run = mlflow.start_run(run_id=run_id)

            if active_run and active_run.info.status == "FAILED":
                client = mlflow.tracking.MlflowClient()
                client.update_run(active_run.info.run_id, status="RUNNING")

            if torch.is_tensor(image):
                arr = image.detach().cpu().numpy()
                if arr.ndim == 2:
                    arr = arr.astype(np.float32)
                elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                    arr = arr.transpose(1, 2, 0)

                if arr.ndim == 3 and arr.shape[-1] == 1:
                    arr = arr[..., 0]
                if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                    arr = np.clip(arr, 0.0, 1.0)
                    arr = (arr * 255).astype(np.uint8)
                elif arr.ndim == 2:
                    arr = np.clip(arr, 0.0, 1.0)
                    arr = (arr * 255).astype(np.uint8)
                
                # 💡 MUDANÇA AQUI: usar `key` em vez de `artifact_file`
                mlflow.log_image(arr, key=name, step=step)
            else:
                mlflow.log_image(image, key=name, step=step)
        except Exception:
            pass

    def _log_metric_value(self, pl_module: pl.LightningModule, name: str, value, step: int | None = None) -> None:
        tb = self._tensorboard_backend(pl_module)
        mlflow_logger = self._mlflow_logger(pl_module)
        if mlflow_logger is not None:
            try:
                mlflow_logger.log_metrics({name: float(value)}, step=step)
            except Exception:
                pass
            return
        if tb is not None:
            tb.add_scalar(name, float(value), step)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        tb = self._tensorboard_backend(pl_module)
        log_dir = self._logger_base_dir(trainer, fallback_dir=getattr(getattr(pl_module, "cfg", None), "logging", None).log_dir if getattr(pl_module, "cfg", None) is not None and hasattr(getattr(pl_module, "cfg", None), "logging") else "tb_logs")
        os.makedirs(log_dir, exist_ok=True)

        if tb is not None and os.path.exists(self.config_path):
            shutil.copy(self.config_path, os.path.join(log_dir, "config.yaml"))
            with open(self.config_path) as f:
                config_content = f.read()
            tb.add_text("config", config_content, global_step=0)
            tb.add_text("model_summary", str(pl_module), global_step=0)
        elif os.path.exists(self.config_path):
            dst = os.path.join(log_dir, "config.yaml")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(self.config_path, dst)
            self._log_mlflow_artifact(pl_module, dst, artifact_path="config")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.histogram_logging_frequency > 0 and batch_idx % self.histogram_logging_frequency == 0:
            tb = self._tensorboard_backend(pl_module)
            mlflow_logger = self._mlflow_logger(pl_module)

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
            metrics_to_log = {}

            for name, param in hists.items():
                # 1. Log dos Pesos (Histograma e Norma L2)
                if tb is not None:
                    tb.add_histogram(f"weights/{name}", param.detach().cpu(), global_step=step)

                weight_norm = torch.linalg.vector_norm(param).item()
                metrics_to_log[f"norms_weight/{name}"] = float(weight_norm)

                # 2. Log dos Gradientes (Histograma e Norma L2)
                if param.grad is not None:
                    if tb is not None:
                        tb.add_histogram(f"grads/{name}", param.grad.detach().cpu(), global_step=step)

                    grad_norm = torch.linalg.vector_norm(param.grad).item()
                    metrics_to_log[f"norms_grad/{name}"] = float(grad_norm)

            if mlflow_logger is not None:
                try:
                    mlflow_logger.log_metrics(metrics_to_log, step=step)
                except Exception:
                    pass

            if tb is not None:
                for metric_name, value in metrics_to_log.items():
                    tb.add_scalar(metric_name, value, global_step=step)
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
        print("VAL BUFFER LEN", len(getattr(pl_module, "val_audio_buffer", [])))
        print("TEST BUFFER LEN", len(getattr(pl_module, "test_audio_buffer", [])))
        local_path = getattr(getattr(pl_module, "cfg", None), "test", None)
        local_path = getattr(local_path, "save_dir", None) if local_path is not None else None
        if local_path is None:
            local_path = "test_results/samples"
        os.makedirs(local_path, exist_ok=True)
        self.save_from_buffer(pl_module, "val_audio_buffer", "val_binaries", local_path=local_path)

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

        if local_path is None and (not hasattr(trainer.logger, "experiment") or not hasattr(trainer.logger.experiment, "add_audio")):
            local_path = getattr(pl_module.cfg.test, "save_dir", "test_results/samples")
            os.makedirs(local_path, exist_ok=True)

        # 1. Salva amostras de áudio
        self.save_from_buffer(
            pl_module,
            "test_audio_buffer",
            "test_binaries",
            local_path=local_path,
            sample_limit=pl_module.cfg.test.trajectory_n_audios,
        )

        # 2. Salva métricas (CSV/Histogramas)
        self.save_test_metrics(pl_module, local_path=local_path, trainer=trainer)

    # ---------- AUX METHODS ----------
    def save_from_buffer(
        self,
        pl_module: pl.LightningModule,
        buf_name: str,
        tb_preffix: str,
        local_path: str | None = None,
        sample_limit: int | None = None,
    ) -> None:
        if local_path is None:
            cfg_test = getattr(getattr(pl_module, "cfg", None), "test", None)
            local_path = getattr(cfg_test, "save_dir", None) if cfg_test is not None else None
        if local_path is None:
            local_path = "test_results/samples"
        os.makedirs(local_path, exist_ok=True)

        buf = getattr(pl_module, buf_name, [])
        if not buf:
            return

        tb = self._tensorboard_backend(pl_module)

        sr = pl_module.audio_sample_rate
        step = pl_module.global_step

        samples_to_log = buf[: self.n_samples if sample_limit is None else sample_limit]

        for i, buf_data in enumerate(samples_to_log):
            suffix = f"sample_{i}"
            cache_key = f"{tb_preffix}_{suffix}"
            metrics = buf_data.get("metrics", {})
            trajectory_audios = buf_data.get("trajectory_audios")
            clean, noisy, enhanced = buf_data["clean"], buf_data["noisy"], buf_data["enhanced"]

            if cache_key not in self._static_cache:
                with torch.no_grad():
                    def to_consistent_wav(wav):
                        return wav.cpu()

                    c_wav = to_consistent_wav(clean)
                    n_wav = to_consistent_wav(noisy)
                    diff_wav = c_wav - n_wav

                    self._static_cache[cache_key] = {
                        "clean": c_wav, "noisy": n_wav, "clean_noisy": diff_wav,
                        "wave_clean": self._plot_waveform_to_tensor(c_wav, "Clean"),
                        "wave_noisy": self._plot_waveform_to_tensor(n_wav, "Noisy"),
                        "wave_diff": self._plot_waveform_to_tensor(diff_wav, "Clean - Noisy"),
                    }

            static_data = self._static_cache[cache_key]
            clean, noisy = static_data["clean"], static_data["noisy"]

            if tb is not None:
                for group_metrics in metrics.values():
                    for m_name, m_val in group_metrics.items():
                        try:
                            val_to_log = float(m_val)
                            tb.add_scalar(f"{tb_preffix}_{suffix}/{m_name}", val_to_log, step)
                        except (ValueError, TypeError):
                            pass
            else:
                mlflow_logger = self._mlflow_logger(pl_module)
                if mlflow_logger is not None:
                    for group_metrics in metrics.values():
                        for m_name, m_val in group_metrics.items():
                            try:
                                value = float(m_val)
                                mlflow_logger.log_metrics({f"{tb_preffix}_{suffix}/{m_name}": value}, step=step)
                            except (ValueError, TypeError):
                                pass

            self._log_rich_audio_data(
                pl_module.cfg.audio,
                tb,
                static_data,
                enhanced,
                trajectory_audios,
                step,
                sr,
                suffix=suffix,
                tb_preffix=tb_preffix,
                local_path=local_path,
                pl_module=pl_module,
            )

        buf.clear()
        plt.close('all')
        gc.collect()

    @staticmethod
    def to_absolute_db(wav, mel_trans):
        mel_spec = mel_trans.to(wav.device)(wav.flatten().unsqueeze(0))
        db_spec = 10.0 * torch.log10(torch.clamp(mel_spec, min=1e-8))
        return db_spec

    def _log_rich_audio_data(
        self,
        cfg,
        tb,
        static_data: dict,
        enhanced: torch.Tensor,
        trajectory_audios: dict[str, torch.Tensor] | None,
        step: int,
        sr: int,
        suffix: str,
        tb_preffix: str,
        local_path: str = None,
        pl_module: pl.LightningModule | None = None,
    ) -> None:
        # torchaudio.save espera tensores 1D ou 2D. Se vier (1, 1, T), removemos o batch.
        if enhanced.ndim == 3: enhanced = enhanced.squeeze(0)

        def compute_seg_snr(clean, enhanced, frame_len=512, hop_len=256):
            """
            Calcula o Segmental SNR entre o sinal limpo e o melhorado.
            """
            # Garante que são 1D
            clean = clean.flatten()
            enhanced = enhanced.flatten()
            
            # Cria os frames (janelas)
            # clean.unfold(dim, size, step)
            clean_frames = clean.unfold(0, frame_len, hop_len)
            enhanced_frames = enhanced.flatten().unfold(0, frame_len, hop_len)
            
            # Calcula a energia do sinal e do ruído (erro) por frame
            signal_power = torch.sum(clean_frames**2, dim=1)
            noise_power = torch.sum((clean_frames - enhanced_frames)**2, dim=1)
            
            # Adiciona um epsilon para evitar log(0)
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-8) + 1e-8)
            
            # Clamping: Prática padrão para evitar que silêncio ou outliers dominem a métrica
            snr = torch.clamp(snr, min=-10, max=35)
            
            return torch.mean(snr).item()
        
        mlflow_logger = self._mlflow_logger(pl_module) if pl_module is not None else None

        if local_path:
            sample_dir = os.path.join(local_path, suffix)
            os.makedirs(sample_dir, exist_ok=True)
            torchaudio.save(os.path.join(sample_dir, "clean.wav"), static_data['clean'].cpu(), sr)
            torchaudio.save(os.path.join(sample_dir, "noisy.wav"), static_data['noisy'].cpu(), sr)
            torchaudio.save(os.path.join(sample_dir, "enhanced.wav"), enhanced.cpu(), sr)
            if trajectory_audios:
                for t_name, t_audio in trajectory_audios.items():
                    wav = t_audio.detach().cpu()
                    if wav.ndim == 1:
                        wav = wav.unsqueeze(0)
                    safe_name = t_name.replace(".", "p")
                    torchaudio.save(os.path.join(sample_dir, f"enhanced_{safe_name}.wav"), wav, sr)

            if mlflow_logger is not None:
                for artifact_name in ["clean.wav", "noisy.wav", "enhanced.wav"]:
                    self._log_mlflow_artifact(pl_module, os.path.join(sample_dir, artifact_name), artifact_path=f"{tb_preffix}_{suffix}")
                if trajectory_audios:
                    for t_name, t_audio in trajectory_audios.items():
                        safe_name = t_name.replace(".", "p")
                        self._log_mlflow_artifact(
                            pl_module,
                            os.path.join(sample_dir, f"enhanced_{safe_name}.wav"),
                            artifact_path=f"{tb_preffix}_{suffix}",
                        )

        # --- Audio & Waveform Plots (Logger) ---
        enhanced_noisy = enhanced - static_data['noisy']
        enhanced_clean = enhanced - static_data['clean']


        if tb is not None:
            tb.add_audio(f"{tb_preffix}_{suffix}/audio_clean", static_data['clean'].flatten().unsqueeze(0), step, sample_rate=sr)
            tb.add_audio(f"{tb_preffix}_{suffix}/audio_noisy", static_data['noisy'].flatten().unsqueeze(0), step, sample_rate=sr)
            tb.add_audio(f"{tb_preffix}_{suffix}/audio_enhanced", enhanced.flatten().unsqueeze(0), step, sample_rate=sr)
            tb.add_audio(f"{tb_preffix}_{suffix}/audio_enhanced_noisy", enhanced_noisy.flatten().unsqueeze(0), step, sample_rate=sr)
            tb.add_audio(f"{tb_preffix}_{suffix}/audio_enhanced_clean", enhanced_clean.flatten().unsqueeze(0), step, sample_rate=sr)
            tb.add_audio(f"{tb_preffix}_{suffix}/audio_clean_noisy", static_data['clean_noisy'].flatten().unsqueeze(0), step, sample_rate=sr)
            if trajectory_audios:
                for t_name, t_audio in trajectory_audios.items():
                    tb.add_audio(
                        f"{tb_preffix}_{suffix}/audio_enhanced_{t_name}",
                        t_audio.flatten().unsqueeze(0),
                        step,
                        sample_rate=sr,
                    )

            tb.add_image(f"{tb_preffix}_{suffix}/wave_clean", static_data['wave_clean'], step)
            tb.add_image(f"{tb_preffix}_{suffix}/wave_noisy", static_data['wave_noisy'], step)
            tb.add_image(f"{tb_preffix}_{suffix}/wave_enhanced", self._plot_waveform_to_tensor(enhanced, "Enhanced"), step)
            tb.add_image(f"{tb_preffix}_{suffix}/wave_enhanced_noisy", self._plot_waveform_to_tensor(enhanced_noisy, "Enhanced - Noisy"), step)
            tb.add_image(f"{tb_preffix}_{suffix}/wave_enhanced_clean", self._plot_waveform_to_tensor(enhanced_clean, "Enhanced - Clean"), step)
            tb.add_image(f"{tb_preffix}_{suffix}/wave_clean_noisy", static_data['wave_diff'], step)
        elif mlflow_logger is not None:
            for label, wav in {
                "audio_clean": static_data['clean'],
                "audio_noisy": static_data['noisy'],
                "audio_enhanced": enhanced,
                "audio_enhanced_noisy": enhanced_noisy,
                "audio_enhanced_clean": enhanced_clean,
                "audio_clean_noisy": static_data['clean_noisy'],
            }.items():
                tmp_path = os.path.join(local_path or "test_results", f"{tb_preffix}_{suffix}_{label}_step{step}.wav") if local_path else os.path.join("test_results", f"{tb_preffix}_{suffix}_{label}_step{step}.wav")
                os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
                torchaudio.save(tmp_path, wav.flatten().unsqueeze(0).cpu(), sr)
                try:
                    self._log_mlflow_artifact(pl_module, tmp_path, artifact_path=f"{tb_preffix}_{suffix}_step{step}")
                finally:
                    if local_path is None:
                        try:
                            os.remove(tmp_path)
                        except FileNotFoundError:
                            pass

            if trajectory_audios:
                for t_name, t_audio in trajectory_audios.items():
                    wav = t_audio.detach().cpu()
                    if wav.ndim == 1:
                        wav = wav.unsqueeze(0)
                    safe_name = t_name.replace(".", "p")
                    tmp_path = os.path.join(local_path or "test_results", f"{tb_preffix}_{suffix}_enhanced_{safe_name}_step{step}.wav") if local_path else os.path.join("test_results", f"{tb_preffix}_{suffix}_enhanced_{safe_name}_step{step}.wav")
                    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
                    torchaudio.save(tmp_path, wav, sr)
                    try:
                        self._log_mlflow_artifact(pl_module, tmp_path, artifact_path=f"{tb_preffix}_{suffix}_enhanced_{safe_name}_step{step}")
                    finally:
                        if local_path is None:
                            try:
                                os.remove(tmp_path)
                            except FileNotFoundError:
                                pass

            self._log_mlflow_image(pl_module, static_data['wave_clean'], f"{tb_preffix}_{suffix}/wave_clean", step)
            self._log_mlflow_image(pl_module, static_data['wave_noisy'], f"{tb_preffix}_{suffix}/wave_noisy", step)
            self._log_mlflow_image(pl_module, self._plot_waveform_to_tensor(enhanced, "Enhanced"), f"{tb_preffix}_{suffix}/wave_enhanced", step)
            self._log_mlflow_image(pl_module, self._plot_waveform_to_tensor(enhanced_noisy, "Enhanced - Noisy"), f"{tb_preffix}_{suffix}/wave_enhanced_noisy", step)
            self._log_mlflow_image(pl_module, self._plot_waveform_to_tensor(enhanced_clean, "Enhanced - Clean"), f"{tb_preffix}_{suffix}/wave_enhanced_clean", step)
            self._log_mlflow_image(pl_module, static_data['wave_diff'], f"{tb_preffix}_{suffix}/wave_clean_noisy", step)
            

        mel_trans = self._get_mel_trans(cfg)
        spec_clean = self.to_absolute_db(static_data['clean'], mel_trans)
        spec_noisy = self.to_absolute_db(static_data['noisy'], mel_trans)
        spec_diff = self.to_absolute_db(static_data['clean_noisy'], mel_trans)
        spec_enhanced = self.to_absolute_db(enhanced, mel_trans)
        spec_enhanced_noisy = self.to_absolute_db(enhanced_noisy, mel_trans)
        spec_enhanced_clean = self.to_absolute_db(enhanced_clean, mel_trans)

        v_max = spec_clean.max().item()
        v_min = v_max - 80.0

        stft_mag_clean, stft_phase_clean = self._stft_to_images(
            static_data['clean'], n_fft=cfg.fft_size, hop_length=cfg.hop_length
        )
        stft_mag_noisy, stft_phase_noisy = self._stft_to_images(
            static_data['noisy'], n_fft=cfg.fft_size, hop_length=cfg.hop_length
        )
        stft_mag_enhanced, stft_phase_enhanced = self._stft_to_images(
            enhanced, n_fft=cfg.fft_size, hop_length=cfg.hop_length
        )

        for tag, spec in [
            ("clean", spec_clean),
            ("noisy", spec_noisy),
            ("enhanced", spec_enhanced),
            ("enhanced_noisy", spec_enhanced_noisy),
            ("enhanced_clean", spec_enhanced_clean),
            ("clean_noisy", spec_diff),
        ]:
            img = (spec - v_min) / (v_max - v_min + 1e-8)
            img = torch.clamp(img, 0, 1)
            if tb is not None:
                tb.add_image(f"{tb_preffix}_{suffix}/spec_{tag}", img, step)
            elif mlflow_logger is not None:
                self._log_mlflow_image(pl_module, img, f"{tb_preffix}_{suffix}/spec_{tag}", step)

        for tag, image in [
            ("stft_mag_clean", stft_mag_clean),
            ("stft_mag_noisy", stft_mag_noisy),
            ("stft_mag_enhanced", stft_mag_enhanced),
            ("stft_phase_clean", stft_phase_clean),
            ("stft_phase_noisy", stft_phase_noisy),
            ("stft_phase_enhanced", stft_phase_enhanced),
        ]:
            if tb is not None:
                tb.add_image(f"{tb_preffix}_{suffix}/{tag}", image, step)
            elif mlflow_logger is not None:
                self._log_mlflow_image(pl_module, image, f"{tb_preffix}_{suffix}/{tag}", step)

        # MSE
        mse_rec = F.mse_loss(static_data['clean_noisy'], enhanced_noisy.flatten())
        mse_noise = F.mse_loss(static_data['clean'], enhanced.flatten())

        # Energy Ratio
        energy_clean = torch.sum(static_data['clean']**2) + 1e-8
        energy_enhanced = torch.sum(enhanced**2)
        energy_ratio = (energy_enhanced / energy_clean).item()
    
        # Alinhamento de fase
        clean_flat = static_data['clean'].flatten()
        enhanced_flat = enhanced.flatten()
        correlation = torch.corrcoef(torch.stack([clean_flat, enhanced_flat]))[0, 1].item()

        # SegSNR
        seg_snr = compute_seg_snr(static_data['clean'], enhanced)

        metrics_to_log = {
            "seg_snr": seg_snr,
            "energy_ratio": energy_ratio,
            "correlation": correlation,
            "mse_reconstruction": mse_rec,
            "mse_noise_extraction": mse_noise,
        }
        if tb is not None:
            for m_name, m_val in metrics_to_log.items():
                tb.add_scalar(f"{tb_preffix}_{suffix}/{m_name}", m_val, step)
        elif mlflow_logger is not None:
            for m_name, m_val in metrics_to_log.items():
                mlflow_logger.log_metrics({f"{tb_preffix}_{suffix}/{m_name}": float(m_val)}, step=step)

    def save_test_metrics(self, pl_module: pl.LightningModule, local_path: str = None, trainer: pl.Trainer = None) -> None:
        tb = self._tensorboard_backend(pl_module)

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

        # 2. Cria o DataFrame completo (inclui colunas numéricas e categóricas, ex: metadados)
        df = pd.DataFrame(all_metrics)

        # 3. Salva o CSV como texto no TensorBoard
        csv_text = df.to_csv(index=False)
        logger = getattr(trainer, "logger", None)
        base_dir = local_path or getattr(logger, "log_dir", None) or "test_results"
        os.makedirs(base_dir, exist_ok=True)

        if tb:
            tb.add_text("test/individual_metrics", f"```csv\n{csv_text}\n```", global_step=pl_module.global_step)
        elif self._mlflow_logger(pl_module) is not None:
            csv_path = os.path.join(base_dir, "test_metrics.csv")
            df.to_csv(csv_path, index=False)
            self._log_mlflow_artifact(pl_module, csv_path, artifact_path="test")

        save_path = os.path.join(base_dir, "test_metrics.csv")
        df.to_csv(save_path, index=False)
        # 4. Limpa buffer para liberar memória
        pl_module.test_metrics_buffer.clear()
