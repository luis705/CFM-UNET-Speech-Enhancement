from einops import rearrange
import pytorch_lightning as pL
import torch
import torch.nn.functional as F
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio, deep_noise_suppression_mean_opinion_score

from src.cfm.loss import MRSTFTLoss, cfm_loss
from src.cfm.solver import ode_solve
from src.data.transforms import preprocess_item, signal_to_stft, stft_to_waveform
from src.model.unet import CFMUnet
from src.utils.metrics import compute_estoi, compute_wb_pesq
from src.utils.settings import Settings


class CFMModule(pL.LightningModule):
    def __init__(self, cfg: Settings):
        super().__init__()
        # Salva as configurações para o logger (TensorBoard/WandB) e checkpoints
        self.save_hyperparameters(cfg.model_dump())
        self.cfg = cfg

        self.model = CFMUnet(
            in_channels=cfg.data.input_channels,
            out_channels=cfg.data.output_channels,
            time_emb_dim=cfg.model.time_emb_dim,
            projection_dim=cfg.model.projection_dim,
            n_blocks=cfg.model.n_blocks,
            compression_factor=cfg.model.compression_factor,
            channels_factor=cfg.model.channels_factor,
            num_heads=cfg.model.num_heads,
            ff_mult=cfg.model.ff_mult,
            n_groups=cfg.model.n_groups,
            gradient_checkpointing=cfg.model.gradient_checkpointing,
        )

        # ODE
        self.ode_val_steps = cfg.ode.val_steps
        self.ode_test_steps = cfg.ode.test_steps

        # Audio
        self.audio_length_seconds = cfg.audio.audio_length_seconds
        self.audio_sample_rate = cfg.audio.sample_rate
        self.fft_size = cfg.audio.fft_size
        self.hop_length = cfg.audio.hop_length
        self.window_size = cfg.audio.window_size
        self.alpha = cfg.audio.alpha

        # Optimizer
        self.lr = cfg.training.optim.lr
        self.weight_decay = cfg.training.optim.weight_decay
        self.pct_start = cfg.training.optim.pct_start

        # Losses
        self.lambda_cfm = cfg.training.losses.cfm.lambda_value
        self.snr_min_val = cfg.training.losses.cfm.snr_min_val
        self.mrstft_lambda_max = cfg.training.losses.mrstft.lambda_max
        self.mrstft_start_epoch = cfg.training.losses.mrstft.start_epoch
        self.mrstft_end_epoch = cfg.training.losses.mrstft.end_epoch

        # Logging
        self.validation_slow_metrics_pct = cfg.training.validation_slow_metrics_pct
        self.n_audios_to_log = cfg.logging.n_audios
        self.val_audio_buffer = []
        self.test_audio_buffer = []
        self.test_metrics_buffer = []
        self._baseline_cache = {}

        self.mrstft_loss_func = MRSTFTLoss(
            cfg.training.losses.mrstft.fft_sizes,
            cfg.training.losses.mrstft.hop_sizes,
            cfg.training.losses.mrstft.win_sizes,
        )

        # Waveform input: (Batch, Channels, Time)
        self.example_input_array = torch.zeros(1, 1, self.audio_length_seconds * self.audio_sample_rate)

    def training_step(self, batch, batch_idx):
        noisy_wav, clean_wav = batch

        # Preprocessamento na GPU
        _, clean_wav_p, noisy_stft, clean_stft, mean, std = preprocess_item(
            noisy_wav,
            clean_wav,
            pad_crop=True,
            length=self.audio_length_seconds * self.audio_sample_rate,
            fft_size=self.fft_size,
            hop_length=self.hop_length,
            window_size=self.window_size,
            alpha=self.alpha,
        )

        cfm_l, v_pred = self.__interpolate_and_generate_cfm_loss(noisy_stft, clean_stft)

        # Cálculo dinâmico do peso da MR-STFT
        lambda_mrstft = self._get_lambda_mrstft(batch_idx)

        if lambda_mrstft > 0:
            est_clean_stft = noisy_stft + v_pred
            est_clean_wav = stft_to_waveform(
                est_clean_stft,
                length=self.audio_length_seconds * self.audio_sample_rate,
                fft_size=self.fft_size,
                hop_length=self.hop_length,
                window_size=self.window_size,
                alpha=self.alpha,
                mean=mean,
                std=std,
            )
            # MRSTFTLoss espera (B, T)
            mr_l = self.mrstft_loss_func(est_clean_wav, clean_wav_p.squeeze(1))

        else:
            mr_l = torch.tensor(0.0, device=self.device)

        loss = self.lambda_cfm * cfm_l + lambda_mrstft * mr_l

        self.log("train/loss", loss.detach(), prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss_cfm", cfm_l.detach(), on_step=True, on_epoch=True)
        self.log("train/loss_mrstft", mr_l.detach(), on_step=True, on_epoch=True)
        self.log("train/lambda_mrstft", torch.tensor(lambda_mrstft, device=self.device), on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        noisy_wav, clean_wav = batch

        # 1. Preprocessamento padrão (Crop para cálculo da Loss)
        # Isso garante que a loss de validação seja comparável com a de treino
        _, clean_wav_p, noisy_stft_crop, clean_stft_crop, mean, std = preprocess_item(
            noisy_wav,
            clean_wav,
            pad_crop=True,
            length=self.audio_length_seconds * self.audio_sample_rate,
            fft_size=self.fft_size,
            hop_length=self.hop_length,
            window_size=self.window_size,
            alpha=self.alpha,
        )

        # Cálculo da Loss (Sempre feito para monitorar convergência)
        cfm_l, v_pred = self.__interpolate_and_generate_cfm_loss(noisy_stft_crop, clean_stft_crop)

        lambda_mrstft = self._get_lambda_mrstft(batch_idx)

        if lambda_mrstft > 0:
            est_clean_stft = noisy_stft_crop + v_pred
            est_clean_wav = stft_to_waveform(
                est_clean_stft,
                length=self.audio_length_seconds * self.audio_sample_rate,
                fft_size=self.fft_size,
                hop_length=self.hop_length,
                window_size=self.window_size,
                alpha=self.alpha,
                mean=mean,
                std=std,
            )
            mr_l = self.mrstft_loss_func(est_clean_wav, clean_wav_p.squeeze(1))

        else:
            mr_l = torch.tensor(0.0, device=self.device)

        loss = self.lambda_cfm * cfm_l + lambda_mrstft * mr_l

        self.log("val/loss", loss.detach(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss_cfm", cfm_l.detach(), on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss_mrstft", mr_l.detach(), on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/lambda_mrstft", torch.tensor(lambda_mrstft, device=self.device), on_step=False, on_epoch=True, sync_dist=True)

        # 2. Métricas lentas (OLA + PESQ) em uma fração dos dados
        num_val_batches = self.trainer.num_val_batches[0]
        limit = max(1, int(num_val_batches * self.validation_slow_metrics_pct))

        if batch_idx < limit:
            self.process_and_log_metrics(noisy_wav, clean_wav, "val", self.val_audio_buffer, batch_idx)

        return loss

    def _get_lambda_mrstft(self, batch_idx: int) -> float:
        """Calcula o peso da loss MR-STFT com rampa linear a cada passo."""
        if self.trainer is None:
            return 0.0

        current_epoch = self.current_epoch
        start = self.mrstft_start_epoch
        end = self.mrstft_end_epoch

        if current_epoch < start:
            return 0.0
        if current_epoch >= end:
            return self.mrstft_lambda_max

        # Rampa linear entre épocas incluindo progressão intra-época (passos)
        steps_per_epoch = self.trainer.num_training_batches
        if steps_per_epoch == 0 or steps_per_epoch == float("inf"):
            return self.mrstft_lambda_max * (current_epoch - start) / (end - start)

        t = (current_epoch - start) / (end - start)
        t += (batch_idx / steps_per_epoch) / (end - start)

        return self.mrstft_lambda_max * max(0.0, min(1.0, t))

    def test_step(self, batch, batch_idx):
        noisy_wav, clean_wav = batch
        metrics_list = self.process_and_log_metrics(noisy_wav, clean_wav, "test", self.test_audio_buffer, batch_idx)
        self.test_metrics_buffer.extend(metrics_list)

    def forward(self, x, n_steps: int = None):
        n_steps = n_steps or self.ode_test_steps
        return self.process_ola(x, n_steps=n_steps)

    def process_and_log_metrics(self, noisy_wav: torch.Tensor, clean_wav: torch.Tensor, log_suffix: str, buffer: list, batch_idx: int):
        # Roda o OLA no áudio completo
        n_steps = self.ode_val_steps if log_suffix == "val" else self.ode_test_steps
        enhanced_wav = self.process_ola(noisy_wav, n_steps=n_steps)

        # Para o MSE de áudio, precisamos do STFT do áudio realçado
        enhanced_stft = signal_to_stft(enhanced_wav, self.fft_size, self.hop_length, self.window_size, self.alpha)

        # Garante o alinhamento caso haja pequenas diferenças residuais do OLA
        T_target = enhanced_wav.shape[-1]
        clean_wav_aligned = clean_wav[..., :T_target]
        clean_stft = signal_to_stft(clean_wav_aligned, self.fft_size, self.hop_length, self.window_size, self.alpha)

        # Cálculo individual por áudio no batch para granularidade total
        B = noisy_wav.shape[0]
        individual_metrics = []

        dnsmos = deep_noise_suppression_mean_opinion_score(enhanced_wav, self.audio_sample_rate, personalized=True)
        dnsmos_noisy = deep_noise_suppression_mean_opinion_score(noisy_wav, self.audio_sample_rate, personalized=True)
        dnsmos_clean = deep_noise_suppression_mean_opinion_score(clean_wav_aligned, self.audio_sample_rate, personalized=True)

        for i in range(B):
            s_enhanced = enhanced_wav[i]
            s_clean = clean_wav_aligned[i]
            s_noisy = noisy_wav[i]

            # --- Baseline Metrics (Cached) ---
            # Identificador único por split e amostra para evitar colisões
            sample_key = (log_suffix, batch_idx, i)
            
            if sample_key not in self._baseline_cache:
                # DNSMOS para noisy e clean (cálculo pesado feito apenas 1 vez)
                
                self._baseline_cache[sample_key] = {
                    "PESQ_NOISY": float(compute_wb_pesq(s_noisy, s_clean)),
                    "STOI_NOISY": float(compute_estoi(s_noisy, s_clean)),
                    "SI_SDR_NOISY": float(scale_invariant_signal_distortion_ratio(s_noisy, s_clean)),
                }

            # Métricas individuais (PESQ e STOI geralmente operam em amostras únicas)
            pesq = float(compute_wb_pesq(s_enhanced, s_clean))
            stoi = float(compute_estoi(s_enhanced, s_clean))

            si_sdr_val = float(scale_invariant_signal_distortion_ratio(s_enhanced, s_clean))

            mse = F.mse_loss(enhanced_stft[i : i + 1], clean_stft[i : i + 1]).item()

            res = {
                'sample': {
                    "PESQ": pesq,
                    "STOI": stoi,
                    "SI_SDR": si_sdr_val,
                    "MSE_AUDIO": mse,
                },
                'DNSMOS': {
                    "MOS": dnsmos[i, 0, 0].item(),
                    "SIG": dnsmos[i, 0, 1].item(),
                    "BAK": dnsmos[i, 0, 2].item(),
                    "OVL": dnsmos[i, 0, 3].item(),
                    "CLEAN_MOS": dnsmos_clean[i, 0, 0].item(),
                    "CLEAN_SIG": dnsmos_clean[i, 0, 1].item(),
                    "CLEAN_BAK": dnsmos_clean[i, 0, 2].item(),
                    "CLEAN_OVL": dnsmos_clean[i, 0, 3].item(),
                    "NOISY_MOS": dnsmos_noisy[i, 0, 0].item(),
                    "NOISY_SIG": dnsmos_noisy[i, 0, 1].item(),
                    "NOISY_BAK": dnsmos_noisy[i, 0, 2].item(),
                    "NOISY_OVL": dnsmos_noisy[i, 0, 3].item(),
                }
            }
            # Adiciona métricas baseadas no cache para a média do batch e buffer de áudio
            res["sample"].update(self._baseline_cache[sample_key])
            individual_metrics.append(res)

            # Salva no buffer de áudio (LoggingCallback) se ainda houver espaço
            if len(buffer) < self.n_audios_to_log:
                buffer.append(
                    {
                        "clean": s_clean.detach().cpu(),
                        "noisy": s_noisy.detach().cpu(),
                        "enhanced": s_enhanced.detach().cpu(),
                        "metrics": res
                    }
                )

        # Log das médias do batch para o Lightning (Scalars do TensorBoard)
        if individual_metrics:
            for outer_key in individual_metrics[0].keys():
                for inner_key in individual_metrics[0][outer_key].keys():
                    avg = sum(m[outer_key][inner_key] for m in individual_metrics) / B
                    self.log(f"{log_suffix}_{outer_key}/{inner_key}", avg, sync_dist=True, prog_bar=True, on_epoch=True)

        return individual_metrics

    @torch.no_grad()
    def process_ola(self, noisy_wav: torch.Tensor, n_steps: int) -> torch.Tensor:
        B, _C, T_orig = noisy_wav.shape
        win_len = int(self.audio_length_seconds * self.audio_sample_rate)
        hop_len = win_len // 2

        # 1. Pad para ser múltiplo do hop e garantir que a última janela caiba
        pad_total = hop_len - (T_orig % hop_len)
        noisy_padded = F.pad(noisy_wav, (0, pad_total + hop_len))
        T_pad = noisy_padded.shape[-1]

        # 2. Chunking (B, C, T_pad) -> (B, C, win_len, n_chunks) -> (B*n_chunks, C, win_len)
        chunks = noisy_padded.unfold(-1, win_len, hop_len)
        n_chunks = chunks.shape[-2]
        chunks = rearrange(chunks, "B C N L -> (B N) C L")

        # 3. Uso do seu preprocess_item (Normaliza e gera STFT por chunk)
        # Passamos noisy duas vezes pois na validação/inferência não temos o clean original do chunk
        _, _, chunks_stft, _, mean, std = preprocess_item(
            chunks,
            chunks,
            pad_crop=False,
            length=win_len,
            fft_size=self.fft_size,
            hop_length=self.hop_length,
            window_size=self.window_size,
            alpha=self.alpha,
        )

        # 4. Modelo (ODE)
        enhanced_stft_chunks = ode_solve(self.model, chunks_stft, n_steps=n_steps)

        # 5. iSTFT + Desnormalização Local
        # Como passamos o mean/std do chunk, o áudio já sai na escala correta para somar
        enhanced_chunks = stft_to_waveform(
            enhanced_stft_chunks,
            length=win_len,
            fft_size=self.fft_size,
            hop_length=self.hop_length,
            window_size=self.window_size,
            alpha=self.alpha,
            mean=mean,
            std=std,
        )

        # 6. Reconstrução Eficiente com Fold
        # Aplicamos a janela de Hann para crossfade
        window = torch.hann_window(win_len, device=self.device).view(1, 1, win_len)
        enhanced_chunks = enhanced_chunks * window

        # Preparamos para o F.fold: (B, C*win_len, n_chunks)
        combined = rearrange(enhanced_chunks, "(B N) C L -> B (C L) N", B=B)

        # Soma os chunks sobrepostos
        output = F.fold(combined, output_size=(1, T_pad), kernel_size=(1, win_len), stride=(1, hop_len))

        # Soma os pesos das janelas para normalizar a amplitude no overlap
        window_ones = window.expand(B, n_chunks, -1).transpose(1, 2)  # (B, win_len, n_chunks)
        weights = F.fold(window_ones, output_size=(1, T_pad), kernel_size=(1, win_len), stride=(1, hop_len))

        return (output / (weights + 1e-8)).squeeze(2)[..., :T_orig]

    def __interpolate_and_generate_cfm_loss(self, noisy_stft: torch.Tensor, clean_stft: torch.Tensor):
        # Get data shapes
        B, _, _, _ = noisy_stft.shape  # (B, C_in, F, T)

        # Sample random time step
        t = torch.rand((B,), device=self.device)
        t_expand = t.view(B, 1, 1, 1)

        # Interpolate:  x_t = (1 - t) * x0  +  t * x1
        x_t = (1 - t_expand) * noisy_stft + t_expand * clean_stft

        # Model input: [noisy_real, noisy_imag, x_t_real, x_t_imag]
        model_input = torch.cat([noisy_stft, x_t], dim=1)  # (B, 4, F, T)

        v_pred = self.model(model_input, t)  # (B, 2, F, T)

        # Calculamos a perda entre o que a rede previu e os dois pontos originais
        weight = torch.clamp(torch.pow(1 - t_expand, 2) / (torch.pow(t_expand, 2) + 1e-5), max=self.snr_min_val)
        return cfm_loss(v_pred=v_pred, x_0=noisy_stft, x_1=clean_stft, weight=weight), v_pred

    def configure_optimizers(self):
        # AdamW é o padrão-ouro para Continuous Flow Matching e U-Nets
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.pct_start,  # Passa 10% do treino aquecendo (Warmup)
            anneal_strategy="cos",  # Cai em formato de curva cosseno
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Atualiza o LR a cada batch, não a cada época
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    from src.data.datamodule import VBDMDDataModule
    from src.utils.settings import get_settings

    CFG = get_settings()
    data_module = VBDMDDataModule(CFG)
    data_module.setup("fit")
    dataloader = data_module.train_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    noisy_cropped, clean_cropped, noisy_stft, clean_stft, mean, std = next(iter(dataloader))
    noisy_cropped, clean_cropped = next(iter(dataloader))  # Agora o dataloader retorna apenas wavs

    # Movemos todos os elementos do batch para o device
    batch = (
        noisy_cropped.to(device),
        clean_cropped.to(device),
        noisy_stft.to(device),
        clean_stft.to(device),
        mean.to(device),
        std.to(device),
    )

    # 1. Instanciar o módulo
    with torch.device(device):
        module = CFMModule(cfg=CFG)
        batch_size = CFG.data.batch_size

        print(f"\n{'=' * 50}")
        print(f"ESTATÍSTICAS DO MODELO (Batch Size: {batch_size})")
        print(f"{'=' * 50}")

        # 3. Contagem de parâmetros
        total_params = sum(p.numel() for p in module.model.parameters())
        trainable_params = sum(p.numel() for p in module.model.parameters() if p.requires_grad)
        print(f"Parâmetros Totais: {total_params / 1e6:.2f}M")
        print(f"Parâmetros Treináveis: {trainable_params / 1e6:.2f}M")

        # 4. Estimativa de VRAM (em MB)
        # - Pesos (Float32 = 4 bytes)
        # - Gradientes (4 bytes)
        # - Otimizador AdamW (8 bytes por parâmetro para os 2 buffers de momentum)
        param_mem = total_params * 4 / (1024**2)
        grad_mem = trainable_params * 4 / (1024**2)
        opt_mem = trainable_params * 8 / (1024**2)

        print(f"Memória fixa (Pesos + Grads + Opt): {param_mem + grad_mem + opt_mem:.2f} MB")
        print("Nota: Ativações não inclusas (variam conforme o input e profundidade)")

        # 5. Teste de Input/Output
        print(f"\nTestando Forward Pass com shape STFT: {noisy_stft.shape}...")
        loss = module.training_step(batch, 0)
        loss.backward()
        max_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Sucesso! Loss calculada: {loss.item():.4f}")
        print(f"VRAM REAL UTILIZADA (Pico de Ativações + Grads): {max_vram_mb:.2f} MB")
        print(f"{'=' * 50}\n")
