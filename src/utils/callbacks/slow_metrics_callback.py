"""Slow metrics callback for validation/test evaluation."""

import hashlib
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from transformers import AutoFeatureExtractor, WavLMForXVector

from src.utils.settings import Settings


class SlowMetricsCallback(pl.Callback):
    """Compute and log expensive objective metrics in validation/test."""

    def __init__(self, cfg: Settings) -> None:
        super().__init__()
        self.cfg = cfg
        self.audio_sample_rate = cfg.audio.sample_rate

        self._stoi_metric = None
        self._pesq_wb_metric = None
        self._pesq_nb_metric = None
        self._spk_feature_extractor = None
        self._spk_model = None
        self._spk_device = torch.device(cfg.evaluation.spk_sim_device)
        self._utmos_predictor = None
        self._utmos_device = torch.device(cfg.evaluation.utmos_device)

        self._val_seen_samples = 0
        self._test_seen_samples = 0

        self._val_values: dict[str, list[float]] = {
            "stoi": [],
            "pesq_wb": [],
            "pesq_nb": [],
            "spk_sim": [],
            "utmos": [],
        }
        self._test_values: dict[str, list[float]] = {
            "stoi": [],
            "pesq_wb": [],
            "pesq_nb": [],
            "spk_sim": [],
            "utmos": [],
        }

        self._val_spk_gt_cache: dict[int, torch.Tensor] = {}
        self._val_spk_gt_cache_loaded = False
        self._val_spk_gt_cache_dirty = False

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        self._validate_config()

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._val_seen_samples = 0
        for key in self._val_values:
            self._val_values[key].clear()
        self._load_val_spk_gt_cache(trainer)

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._test_seen_samples = 0
        for key in self._test_values:
            self._test_values[key].clear()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking or not isinstance(outputs, dict):
            return
        target = outputs.get("slow_target")
        pred = outputs.get("slow_pred")
        if target is None or pred is None:
            return
        self._compute_slow_metrics(trainer, pl_module, target, pred, split="val")

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not isinstance(outputs, dict):
            return
        target = outputs.get("slow_target")
        pred = outputs.get("slow_pred")
        noisy = outputs.get("slow_noisy")
        if target is None or pred is None:
            return
        self._compute_slow_metrics(
            trainer,
            pl_module,
            target,
            pred,
            split="test",
            noisy=noisy,
            outputs=outputs,
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        self._log_epoch_summary(pl_module, split="val")
        self._save_val_spk_gt_cache(trainer)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log_epoch_summary(pl_module, split="test")

    def _validate_config(self) -> None:
        val_pct = float(self.cfg.evaluation.val_slow_metrics_percent)
        test_pct = float(self.cfg.evaluation.test_slow_metrics_percent)
        if not (0.0 <= val_pct <= 1.0):
            raise ValueError("evaluation.val_slow_metrics_percent must be between 0.0 and 1.0")
        if not (0.0 <= test_pct <= 1.0):
            raise ValueError("evaluation.test_slow_metrics_percent must be between 0.0 and 1.0")

    def _maybe_init_backends(self) -> None:
        val_pct = float(self.cfg.evaluation.val_slow_metrics_percent)
        test_pct = float(self.cfg.evaluation.test_slow_metrics_percent)
        if val_pct <= 0.0 and test_pct <= 0.0:
            return

        if self._stoi_metric is None:
            self._stoi_metric = ShortTimeObjectiveIntelligibility(fs=self.audio_sample_rate, extended=False)
        if self._pesq_wb_metric is None:
            self._pesq_wb_metric = PerceptualEvaluationSpeechQuality(fs=self.audio_sample_rate, mode="wb")
        if self._pesq_nb_metric is None:
            self._pesq_nb_metric = PerceptualEvaluationSpeechQuality(fs=8000, mode="nb")
        if self._spk_model is None or self._spk_feature_extractor is None:
            model_id = self.cfg.evaluation.spk_sim_model_id
            self._spk_feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            self._spk_model = WavLMForXVector.from_pretrained(model_id)
            self._spk_model.eval()
            self._spk_model.to(self._spk_device)
        if self._utmos_predictor is None:
            self._utmos_predictor = torch.hub.load(
                self.cfg.evaluation.utmos_hub_repo,
                self.cfg.evaluation.utmos_model,
                trust_repo=True,
            )
            if hasattr(self._utmos_predictor, "to"):
                self._utmos_predictor.to(self._utmos_device)

    def _slow_metric_cache_dir(self, trainer: pl.Trainer) -> str:
        if trainer.logger is not None and hasattr(trainer.logger, "log_dir"):
            return os.path.join(trainer.logger.log_dir, "metrics_cache")
        return os.path.join(self.cfg.test.save_dir, "metrics_cache")

    def _metrics_output_dir(self, trainer: pl.Trainer) -> str:
        if trainer.logger is not None and hasattr(trainer.logger, "log_dir"):
            return os.path.join(trainer.logger.log_dir, "metrics")
        return self.cfg.test.save_dir

    def _val_spk_gt_cache_path(self, trainer: pl.Trainer) -> str:
        return os.path.join(self._slow_metric_cache_dir(trainer), "val_spk_gt_embeddings.pt")

    def _load_val_spk_gt_cache(self, trainer: pl.Trainer) -> None:
        if self._val_spk_gt_cache_loaded:
            return

        cache_path = self._val_spk_gt_cache_path(trainer)
        if os.path.exists(cache_path):
            try:
                loaded = torch.load(cache_path, map_location="cpu")
                if isinstance(loaded, dict):
                    self._val_spk_gt_cache = {int(k): v for k, v in loaded.items() if isinstance(v, torch.Tensor)}
            except Exception:
                self._val_spk_gt_cache = {}
        self._val_spk_gt_cache_loaded = True

    def _save_val_spk_gt_cache(self, trainer: pl.Trainer) -> None:
        if not self._val_spk_gt_cache_dirty:
            return

        cache_path = self._val_spk_gt_cache_path(trainer)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = f"{cache_path}.tmp"
        torch.save(self._val_spk_gt_cache, tmp_path)
        os.replace(tmp_path, cache_path)
        self._val_spk_gt_cache_dirty = False

    def _get_selected_indices(self, split: str, batch_size: int) -> list[tuple[int, int]]:
        if split == "val":
            percent = float(self.cfg.evaluation.val_slow_metrics_percent)
            base_idx = self._val_seen_samples
            self._val_seen_samples += batch_size
        else:
            percent = float(self.cfg.evaluation.test_slow_metrics_percent)
            base_idx = self._test_seen_samples
            self._test_seen_samples += batch_size

        if percent <= 0.0:
            return []
        if percent >= 1.0:
            return [(i, base_idx + i) for i in range(batch_size)]

        selected: list[tuple[int, int]] = []
        for i in range(batch_size):
            sample_idx = base_idx + i
            key = f"{split}:{sample_idx}".encode("utf-8")
            digest = hashlib.blake2b(key, digest_size=8).digest()
            value = int.from_bytes(digest, "big") / float(2**64)
            if value < percent:
                selected.append((i, sample_idx))
        return selected

    def _ensure_spk_model_device(self) -> None:
        model_device = next(self._spk_model.parameters()).device
        if model_device != self._spk_device:
            self._spk_model.to(self._spk_device)

    def _compute_spk_embeddings_batch(self, waves: torch.Tensor) -> torch.Tensor:
        model_input_sr = 16000
        waves = waves.detach().cpu().to(torch.float32)
        if waves.dim() == 3:
            waves = waves.squeeze(1)
        if waves.dim() == 1:
            waves = waves.unsqueeze(0)

        if self.audio_sample_rate != model_input_sr:
            waves = torchaudio.functional.resample(waves, self.audio_sample_rate, model_input_sr)

        wave_list = [w.numpy() for w in waves]
        inputs = self._spk_feature_extractor(
            wave_list,
            sampling_rate=model_input_sr,
            return_tensors="pt",
            padding=True,
        )

        self._ensure_spk_model_device()
        model_device = next(self._spk_model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._spk_model(**inputs)
            emb = outputs.embeddings

        emb = emb / (torch.linalg.vector_norm(emb, dim=-1, keepdim=True) + 1e-12)
        return emb.detach().cpu()

    def _compute_slow_metrics(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        target: torch.Tensor,
        pred: torch.Tensor,
        split: str,
        noisy: torch.Tensor | None = None,
        outputs: dict | None = None,
    ) -> None:
        self._validate_config()
        self._maybe_init_backends()

        sync = split in {"val", "test"}
        values_by_metric = self._val_values if split == "val" else self._test_values
        selected_indices = self._get_selected_indices(split, target.shape[0])

        pl_module.log(
            f"{split}_slow_metrics/selected_count",
            torch.tensor(float(len(selected_indices)), device=pl_module.device),
            on_step=False,
            on_epoch=True,
            sync_dist=sync,
            reduce_fx="sum",
            batch_size=1,
        )

        if not selected_indices:
            return

        local_indices = [local_idx for local_idx, _ in selected_indices]
        sample_indices = [sample_idx for _, sample_idx in selected_indices]
        ref_batch = target[local_indices].detach().cpu().to(torch.float32)
        deg_batch = pred[local_indices].detach().cpu().to(torch.float32)
        noisy_batch = None
        if noisy is not None:
            noisy_batch = noisy[local_indices].detach().cpu().to(torch.float32)

        if ref_batch.dim() == 3:
            ref_batch = ref_batch.squeeze(1)
        if deg_batch.dim() == 3:
            deg_batch = deg_batch.squeeze(1)
        if noisy_batch is not None and noisy_batch.dim() == 3:
            noisy_batch = noisy_batch.squeeze(1)

        n = min(ref_batch.shape[-1], deg_batch.shape[-1])
        if n <= 1:
            return
        ref_batch = ref_batch[:, :n]
        deg_batch = deg_batch[:, :n]
        if noisy_batch is not None:
            noisy_batch = noisy_batch[:, :n]

        sample_slow_rows: list[dict[str, float | None]] = [
            {
                "STOI": None,
                "STOI_CLEAN": None,
                "PESQ_WB": None,
                "PESQ_WB_CLEAN": None,
                "PESQ_NB": None,
                "PESQ_NB_CLEAN": None,
                "SPK_SIM": None,
                "SPK_SIM_CLEAN": None,
                "UTMOS": None,
                "UTMOS_CLEAN": None,
                "STOI_NOISY": None,
                "PESQ_WB_NOISY": None,
                "PESQ_NB_NOISY": None,
                "SPK_SIM_NOISY": None,
                "UTMOS_NOISY": None,
            }
            for _ in sample_indices
        ]

        stoi_values: list[float] = []
        pesq_wb_values: list[float] = []
        pesq_nb_values: list[float] = []
        spk_sim_values: list[float] = []
        utmos_values: list[float] = []

        def _assign_values(col_name: str, values: list[float]) -> None:
            for row, value in zip(sample_slow_rows, values):
                row[col_name] = value

        def _safe_metric_pair(metric_fn, ref_sig: torch.Tensor, deg_sig: torch.Tensor) -> list[float] | None:
            try:
                out = metric_fn(ref_sig, deg_sig)
                return [float(v.item()) for v in out.detach().flatten().cpu()]
            except Exception:
                return None

        def _safe_compute_stoi(ref_sig: torch.Tensor, sig: torch.Tensor) -> list[float] | None:
            return _safe_metric_pair(self._stoi_metric, ref_sig, sig)

        def _safe_compute_pesq_wb(ref_sig: torch.Tensor, sig: torch.Tensor) -> list[float] | None:
            return _safe_metric_pair(self._pesq_wb_metric, ref_sig, sig)

        # PESQ-NB requires resampling to 8kHz.
        ref_nb_batch = None
        deg_nb_batch = None
        noisy_nb_batch = None
        try:
            ref_nb_batch = torchaudio.functional.resample(ref_batch, self.audio_sample_rate, 8000)
            deg_nb_batch = torchaudio.functional.resample(deg_batch, self.audio_sample_rate, 8000)
            if noisy_batch is not None:
                noisy_nb_batch = torchaudio.functional.resample(noisy_batch, self.audio_sample_rate, 8000)
        except Exception:
            ref_nb_batch = None
            deg_nb_batch = None
            noisy_nb_batch = None

        def _safe_compute_pesq_nb(ref_sig: torch.Tensor, sig: torch.Tensor) -> list[float] | None:
            return _safe_metric_pair(self._pesq_nb_metric, ref_sig, sig)

        def _safe_ref_emb_batch() -> torch.Tensor | None:
            try:
                if split == "val":
                    missing_positions: list[int] = []
                    missing_waves: list[torch.Tensor] = []
                    for pos, sample_idx in enumerate(sample_indices):
                        if sample_idx not in self._val_spk_gt_cache:
                            missing_positions.append(pos)
                            missing_waves.append(ref_batch[pos])

                    if missing_waves:
                        missing_tensor = torch.stack(missing_waves, dim=0)
                        missing_emb = self._compute_spk_embeddings_batch(missing_tensor)
                        for pos, emb in zip(missing_positions, missing_emb):
                            sample_idx = sample_indices[pos]
                            self._val_spk_gt_cache[sample_idx] = emb
                            self._val_spk_gt_cache_dirty = True

                    return torch.stack([self._val_spk_gt_cache[sample_idx] for sample_idx in sample_indices], dim=0)

                return self._compute_spk_embeddings_batch(ref_batch)
            except Exception:
                return None

        ref_emb_batch = _safe_ref_emb_batch()

        def _safe_compute_spk_sim(sig: torch.Tensor, signal_type: str) -> list[float] | None:
            if ref_emb_batch is None:
                return None
            try:
                if signal_type == "clean":
                    return [1.0] * ref_emb_batch.shape[0]
                sig_emb_batch = self._compute_spk_embeddings_batch(sig)
                out = F.cosine_similarity(ref_emb_batch, sig_emb_batch, dim=-1)
                return [float(v.item()) for v in out.detach().flatten().cpu()]
            except Exception:
                return None

        def _safe_compute_utmos(sig: torch.Tensor) -> list[float] | None:
            try:
                out = self._utmos_predictor(sig.to(self._utmos_device), sr=self.audio_sample_rate)
                if isinstance(out, torch.Tensor):
                    return [float(v.item()) for v in out.detach().flatten().cpu()]
            except Exception:
                return None
            return None

        signal_order = ["enhanced", "clean", "noisy"]
        suffix_by_signal = {
            "enhanced": "",
            "clean": "_CLEAN",
            "noisy": "_NOISY",
        }
        wb_signals = {
            "enhanced": deg_batch,
            "clean": ref_batch,
            "noisy": noisy_batch,
        }
        nb_signals = {
            "enhanced": deg_nb_batch,
            "clean": ref_nb_batch,
            "noisy": noisy_nb_batch,
        }

        # (metric_name, csv_prefix, aggregate_key, signal_source, safe_fn)
        metric_specs = [
            ("stoi", "STOI", "stoi", "wb", _safe_compute_stoi),
            ("pesq_wb", "PESQ_WB", "pesq_wb", "wb", _safe_compute_pesq_wb),
            ("pesq_nb", "PESQ_NB", "pesq_nb", "nb", _safe_compute_pesq_nb),
            ("spk_sim", "SPK_SIM", "spk_sim", "wb", _safe_compute_spk_sim),
            ("utmos", "UTMOS", "utmos", "wb", _safe_compute_utmos),
        ]

        primary_metric_values: dict[str, list[float]] = {
            "stoi": [],
            "pesq_wb": [],
            "pesq_nb": [],
            "spk_sim": [],
            "utmos": [],
        }

        for _metric_name, row_prefix, agg_key, source_key, safe_fn in metric_specs:
            signal_map = wb_signals if source_key == "wb" else nb_signals
            ref_sig = signal_map.get("clean")
            if ref_sig is None:
                continue

            for signal_type in signal_order:
                sig = signal_map.get(signal_type)
                if sig is None:
                    continue

                if safe_fn is _safe_compute_utmos:
                    values = safe_fn(sig)
                elif safe_fn is _safe_compute_spk_sim:
                    values = safe_fn(sig, signal_type)
                else:
                    values = safe_fn(ref_sig, sig)

                if values is None:
                    continue

                _assign_values(f"{row_prefix}{suffix_by_signal[signal_type]}", values)
                if signal_type == "enhanced":
                    primary_metric_values[agg_key] = values
                    values_by_metric[agg_key].extend(values)

        stoi_values = primary_metric_values["stoi"]
        pesq_wb_values = primary_metric_values["pesq_wb"]
        pesq_nb_values = primary_metric_values["pesq_nb"]
        spk_sim_values = primary_metric_values["spk_sim"]
        utmos_values = primary_metric_values["utmos"]

        if split == "test" and outputs is not None:
            metrics_list = outputs.get("metrics_list")
            if isinstance(metrics_list, list):
                for pos, local_idx in enumerate(local_indices):
                    if 0 <= local_idx < len(metrics_list) and isinstance(metrics_list[local_idx], dict):
                        slow_group = metrics_list[local_idx].setdefault("slow", {})
                        slow_group.update(sample_slow_rows[pos])

        for metric_name, values in {
            "stoi_count": stoi_values,
            "pesq_wb_count": pesq_wb_values,
            "pesq_nb_count": pesq_nb_values,
            "spk_sim_count": spk_sim_values,
            "utmos_count": utmos_values,
        }.items():
            pl_module.log(
                f"{split}_slow_metrics/{metric_name}",
                torch.tensor(float(len(values)), device=pl_module.device),
                on_step=False,
                on_epoch=True,
                sync_dist=sync,
                reduce_fx="sum",
                batch_size=1,
            )

    def _log_epoch_summary(self, pl_module: pl.LightningModule, split: str) -> None:
        values_by_metric = self._val_values if split == "val" else self._test_values
        sync = split in {"val", "test"}

        for metric_name, values in values_by_metric.items():
            if not values:
                continue

            t = torch.tensor(values, device=pl_module.device, dtype=torch.float32)
            mean = t.mean()
            std = t.std(unbiased=False)

            pl_module.log(f"{split}_slow_metrics/{metric_name}", mean, on_step=False, on_epoch=True, sync_dist=sync)
            pl_module.log(f"{split}_slow_metrics/{metric_name}_std", std, on_step=False, on_epoch=True, sync_dist=sync)
            pl_module.log(
                f"{split}_slow_metrics/{metric_name}_count_epoch",
                torch.tensor(float(len(values)), device=pl_module.device),
                on_step=False,
                on_epoch=True,
                sync_dist=sync,
            )