import pytorch_lightning as pL
import torch
import torch.nn.functional as F

from src.cfm.loss import cfm_loss
from src.cfm.solver import ode_solve
from src.data.transforms import stft_to_waveform
from src.model.unet import CFMUnet
from src.utils.metrics import compute_dnsmos, compute_estoi, compute_wb_pesq


class CFMModule(pL.LightningModule):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 2,
        time_emb_dim: int = 128,
        projection_dim: int = 32,
        n_blocks: int = 4,
        compression_factor: int = 2,
        channels_factor: int = 2,
        num_heads: int = 4,
        ff_mult: int = 2,
        n_groups: int = 4,
        gradient_checkpointing: bool = True,
        ode_val_steps: int = 10,
        ode_test_steps: int = 10,
        audio_length_seconds: int = 5,
        audio_sample_rate: int = 16000,
        fft_size: int = 512,
        hop_length: int = 128,
        window_size: int = 256,
        alpha: float = 0.5,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pct_start: float = 0.1,
        validation_slow_metrics_pct: float = 0.1,
        histogram_logging_frequency: int = 100,
        n_audios_to_log: int = 4,
    ):
        super().__init__()
        self.model = CFMUnet(
            in_channels=in_channels,
            out_channels=out_channels,
            time_emb_dim=time_emb_dim,
            projection_dim=projection_dim,
            n_blocks=n_blocks,
            compression_factor=compression_factor,
            channels_factor=channels_factor,
            num_heads=num_heads,
            ff_mult=ff_mult,
            n_groups=n_groups,
            gradient_checkpointing=gradient_checkpointing,
        )

        # ODE Solver
        self.ode_val_steps = ode_val_steps
        self.ode_test_steps = ode_test_steps

        # Audio
        self.audio_length_seconds = audio_length_seconds
        self.audio_sample_rate = audio_sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.window_size = window_size
        self.alpha = alpha

        # Optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.pct_start = pct_start

        # Others
        self.validation_slow_metrics_pct = validation_slow_metrics_pct
        self.save_hyperparameters()
        self.val_audio_buffer = []
        self.histogram_logging_frequency = histogram_logging_frequency
        self.n_audios_to_log = n_audios_to_log

        # (B, C, F, T)
        # Aqui, é o que o forward espera, ou seja, apenas o áudio sujo
        # Por isso fazemos C = in_channels // 2, já que in_channels considera
        # Ambos sujo e limpo
        self.example_input_array = torch.zeros(
            (   1,
                in_channels // 2,
                fft_size // 2 + 1,
                audio_length_seconds * audio_sample_rate // hop_length
            )
        ).to(self.device)

    def training_step(self, batch, batch_idx):
        _, _, noisy_stft, clean_stft, _, _ = batch
        loss, v_pred = self.__interpolate_and_generate_cfm_loss(noisy_stft, clean_stft)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        if batch_idx % self.histogram_logging_frequency == 0:
            tensorboard = self.logger.experiment

            # Logando a distribuição das predições do campo vetorial
            tensorboard.add_histogram("train/v_pred_dist", v_pred.detach().cpu(), global_step=self.global_step)

            for name, param in self.model.named_parameters():
                tensorboard.add_histogram(f"weights/{name}", param.detach().cpu(), global_step=self.global_step)
                if param.grad is not None:
                    tensorboard.add_histogram(f"grads/{name}", param.grad.detach().cpu(), global_step=self.global_step)
            self.logger.experiment.flush()
        return loss

    def validation_step(self, batch, batch_idx):
        noisy_wav, clean_wav, noisy_stft, clean_stft, mean, std = batch
        cfm_loss, _ = self.__interpolate_and_generate_cfm_loss(noisy_stft, clean_stft)

        num_val_batches = self.trainer.num_val_batches[0]
        limit = max(1, int(num_val_batches * self.validation_slow_metrics_pct))
        if batch_idx < limit:
            enhanced_stft = ode_solve(self.model, noisy_stft, n_steps=self.ode_val_steps)
            enhanced_wav = stft_to_waveform(
                    enhanced_stft,
                    length=self.audio_length_seconds * self.audio_sample_rate,
                    fft_size=self.fft_size,
                    hop_length=self.hop_length,
                    window_size=self.window_size,
                    alpha=self.alpha,
                    mean=mean,
                    std=std,
                )
            
            val_mse_audio = F.mse_loss(enhanced_stft, clean_stft)
            pesq = compute_wb_pesq(enhanced_wav, clean_wav)

            self.log("val/pesq", pesq, prog_bar=True, on_step=True, on_epoch=True)
            self.log("val/mse_audio", val_mse_audio, prog_bar=True, on_step=True, on_epoch=True)

            if len(self.val_audio_buffer) < self.n_audios_to_log:
                needed = self.n_audios_to_log - len(self.val_audio_buffer)
                to_take = min(needed, noisy_wav.shape[0])
                for i in range(to_take):
                    self.val_audio_buffer.append((
                        clean_wav[i].detach().cpu(),
                        noisy_wav[i].detach().cpu(),
                        enhanced_wav[i].detach().cpu()
                    ))
        self.log("val/loss_cfm", cfm_loss, prog_bar=True, on_step=True, on_epoch=True)
        return cfm_loss

    def test_step(self, batch, batch_idx):
        _, clean_wav, noisy_stft, _, mean, std = batch

        # 1. Geração com qualidade máxima (50 a 100 passos)
        enhanced_stft = ode_solve(self.model, noisy_stft, n_steps=self.ode_test_steps)

        # 2. Reconstrução do áudio (Transformada Inversa)
        enhanced_wav = stft_to_waveform(
            enhanced_stft,
            length=self.audio_length_seconds * self.audio_sample_rate,
            fft_size=self.fft_size,
            hop_length=self.hop_length,
            window_size=self.window_size,
            alpha=self.alpha,
            mean=mean,
            std=std,
        )

        # 3. Cálculo das métricas oficiais (Exemplo usando torchmetrics)
        pesq_score = compute_wb_pesq(enhanced_wav, clean_wav)
        stoi_score = compute_estoi(enhanced_wav, clean_wav)
        dnsmos_score = compute_dnsmos(enhanced_wav, clean_wav)

        # 4. Logamos as métricas (o Lightning vai tirar a média de todo o dataset de teste)
        self.log("test/PESQ", pesq_score, sync_dist=True)
        self.log("test/STOI", stoi_score, sync_dist=True)
        # self.log("test/DNSMOS", dnsmos_score, sync_dist=True)

        # Opcional: código para salvar o enhanced_wav no disco usando torchaudio.save(...)

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
        return cfm_loss(v_pred=v_pred, x_0=noisy_stft, x_1=clean_stft), v_pred

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

    def forward(self, noisy_stft: torch.Tensor, n_steps: int | None = None):
        n_steps = n_steps or self.ode_test_steps
        return ode_solve(self.model, noisy_stft, n_steps=n_steps)



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
        module = CFMModule(
            in_channels=CFG.data.input_channels,
            out_channels=CFG.data.output_channels,
            time_emb_dim=CFG.model.time_emb_dim,
            projection_dim=CFG.model.projection_dim,
            n_blocks=CFG.model.n_blocks,
            compression_factor=CFG.model.compression_factor,
            channels_factor=CFG.model.channels_factor,
            num_heads=CFG.model.num_heads,
            ff_mult=CFG.model.ff_mult,
            n_groups=CFG.model.n_groups,
            gradient_checkpointing=CFG.model.gradient_checkpointing,
            ode_val_steps=CFG.ode.val_steps,
            ode_test_steps=CFG.ode.test_steps,
            audio_length_seconds=CFG.audio.audio_length_seconds,
            audio_sample_rate=CFG.audio.sample_rate,
            fft_size=CFG.audio.fft_size,
            hop_length=CFG.audio.hop_length,
            window_size=CFG.audio.window_size,
            alpha=CFG.audio.alpha,
            lr=CFG.training.optim.lr,
            weight_decay=CFG.training.optim.weight_decay,
            pct_start=CFG.training.optim.pct_start,
            validation_slow_metrics_pct=CFG.training.validation_slow_metrics_pct,
            n_audios_to_log=CFG.logging.n_audios,
        )
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
