import copy
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class BertConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 512

    # optimization
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    batch_size: int = 32
    num_train_epochs: int = 3
    early_stopping_patience: int = 8

    # imbalance handling
    use_class_weights: bool = True

    # threshold search
    threshold_grid: tuple[float, ...] = tuple(np.round(np.arange(0.10, 0.91, 0.02), 2))

    # device / precision
    force_cpu: bool = False
    fp16: bool = False   # only meaningful on CUDA
    bf16: bool = False   # only meaningful on CUDA

    # speed / experimentation
    logging_steps: int = 100
    sample_size: int | None = None
    num_workers: int = 0

    # misc
    gradient_clip_val: float = 1.0


class BertRepostPredictor:
    """
    MTX-only BERT repost predictor.

    Expected dataframe columns:
        - text
        - label
        - hashtag
    Optional:
        - A_id
        - S_id
        - P_id
    """

    def __init__(self, config: BertConfig | None = None):
        self.config = config or BertConfig()
        self.required_cols = ["text", "label", "hashtag"]
        self.id_cols = ["A_id", "S_id", "P_id"]

        self.device = get_device(force_cpu=self.config.force_cpu)
        logger.info("Using device: %s", self.device)
        logger.info("MPS available: %s", torch.backends.mps.is_available())

        logger.info("Loading tokenizer: %s", self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
        )
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    # Basic checks / dataframe prep


    def _validate_df(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _subsample_if_needed(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.sample_size is None or len(df) <= self.config.sample_size:
            return df

        if self.config.sample_size < 2:
            raise ValueError("sample_size must be >= 2")

        logger.info(
            "Subsampling from %d rows to %d rows for speed",
            len(df),
            self.config.sample_size,
        )

        sampled_parts = []
        total_rows = len(df)
        for label_value, group in df.groupby("label"):
            frac = len(group) / total_rows
            n_label = max(1, int(round(self.config.sample_size * frac)))
            n_label = min(n_label, len(group))
            sampled_parts.append(group.sample(n=n_label, random_state=42))

        sampled = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=42)
        sampled = sampled.head(self.config.sample_size).reset_index(drop=True)
        return sampled

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_df(df)

        out = df.copy()
        out["text"] = out["text"].fillna("").astype(str)
        out["label"] = pd.to_numeric(out["label"], errors="raise").astype(int)

        unique_labels = sorted(out["label"].unique().tolist())
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError(f"Expected binary labels in {{0,1}}, got {unique_labels}")

        out = out.reset_index(drop=True)
        out = self._subsample_if_needed(out)
        return out

    # --------------------------------------------------
    # Dataset / dataloader
    # --------------------------------------------------

    def _tokenize_df(self, df: pd.DataFrame, split_name: str) -> Dataset:
        t0 = time.perf_counter()

        ds = Dataset.from_pandas(
            df[["text", "label"]].reset_index(drop=True),
            preserve_index=False,
        )

        def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
            return self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.config.max_length,
            )

        ds = ds.map(
            tokenize,
            batched=True,
            desc=f"Tokenizing {split_name}",
        )

        ds = ds.rename_column("label", "labels")

        columns = ["input_ids", "attention_mask", "labels"]
        if "token_type_ids" in ds.column_names:
            columns = ["input_ids", "attention_mask", "token_type_ids", "labels"]

        ds.set_format(type="torch", columns=columns)

        logger.info(
            "Prepared %s dataset | rows=%d | time=%.2fs",
            split_name,
            len(df),
            time.perf_counter() - t0,
        )
        return ds

    def _make_dataloader(self, ds: Dataset, train: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=train,
            collate_fn=self.data_collator,
            num_workers=self.config.num_workers,
            pin_memory=False,
        )

    # --------------------------------------------------
    # Model / loss / metrics
    # --------------------------------------------------

    def _make_model(self) -> AutoModelForSequenceClassification:
        logger.info("Loading model: %s", self.config.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=2,
        )
        return model.to(self.device)

    def _compute_class_weights(self, train_labels: pd.Series) -> torch.Tensor | None:
        if not self.config.use_class_weights:
            return None

        counts = train_labels.value_counts().to_dict()
        n_neg = counts.get(0, 0)
        n_pos = counts.get(1, 0)

        if n_neg == 0 or n_pos == 0:
            return None

        total = n_neg + n_pos
        w_neg = total / (2.0 * n_neg)
        w_pos = total / (2.0 * n_pos)

        weights = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=self.device)
        logger.info(
            "Class weights | neg=%.4f | pos=%.4f | counts={0:%d,1:%d}",
            w_neg,
            w_pos,
            n_neg,
            n_pos,
        )
        return weights

    def _softmax_pos_probs(self, logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs[:, 1]

    def _find_best_threshold(
        self,
        y_true: np.ndarray,
        pos_probs: np.ndarray,
    ) -> tuple[float, float]:
        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in self.config.threshold_grid:
            y_pred = (pos_probs >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)

        return best_threshold, float(best_f1)

    def _evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        threshold: float | None = None,
        loss_weights: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        model.eval()

        all_logits = []
        all_labels = []
        total_loss = 0.0
        total_examples = 0

        loss_fct = torch.nn.CrossEntropyLoss(weight=loss_weights)

        with torch.no_grad():
            for batch in dataloader:
                labels = batch["labels"].to(self.device)
                inputs = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k != "labels"
                }

                outputs = model(**inputs)
                logits = outputs.logits
                loss = loss_fct(logits, labels)

                bs = labels.size(0)
                total_loss += loss.item() * bs
                total_examples += bs

                all_logits.append(logits.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

        logits_np = np.concatenate(all_logits, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)
        pos_probs = self._softmax_pos_probs(logits_np)

        if threshold is None:
            threshold, f1 = self._find_best_threshold(labels_np, pos_probs)
        else:
            preds = (pos_probs >= threshold).astype(int)
            f1 = f1_score(labels_np, preds, zero_division=0)

        preds = (pos_probs >= threshold).astype(int)

        return {
            "loss": total_loss / max(1, total_examples),
            "f1": float(f1),
            "threshold": float(threshold),
            "positive_rate_pred": float(preds.mean()),
            "positive_rate_true": float(labels_np.mean()),
            "n_pred_positive": int(preds.sum()),
            "n_true_positive": int(labels_np.sum()),
            "logits": logits_np,
            "labels": labels_np,
            "probs": pos_probs,
            "preds": preds,
        }

    # --------------------------------------------------
    # Core training
    # --------------------------------------------------

    def _fit_and_eval(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        seed: int,
        run_name: str,
    ) -> float:
        run_t0 = time.perf_counter()
        set_global_seed(seed)

        train_df = self._prepare_df(train_df)
        val_df = self._prepare_df(val_df)
        test_df = self._prepare_df(test_df)

        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            raise ValueError(
                f"Empty split detected: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
            )

        if train_df["label"].nunique() < 2:
            raise ValueError("Training split must contain both classes.")
        if val_df["label"].nunique() < 2:
            raise ValueError("Validation split must contain both classes.")
        if test_df["label"].nunique() < 2:
            raise ValueError("Test split must contain both classes.")

        logger.info(
            "[%s | seed=%d] Split sizes | train=%d | val=%d | test=%d",
            run_name,
            seed,
            len(train_df),
            len(val_df),
            len(test_df),
        )

        train_ds = self._tokenize_df(train_df, "train")
        val_ds = self._tokenize_df(val_df, "val")
        test_ds = self._tokenize_df(test_df, "test")

        train_loader = self._make_dataloader(train_ds, train=True)
        val_loader = self._make_dataloader(val_ds, train=False)
        test_loader = self._make_dataloader(test_ds, train=False)

        model = self._make_model()

        train_size = len(train_ds)
        steps_per_epoch = max(1, math.ceil(train_size / self.config.batch_size))
        total_train_steps = steps_per_epoch * self.config.num_train_epochs
        warmup_steps = int(total_train_steps * self.config.warmup_ratio)

        logger.info(
            "[%s | seed=%d] Training config | train_size=%d | batch_size=%d | epochs=%d | steps_per_epoch=%d | total_steps=%d | warmup_steps=%d",
            run_name,
            seed,
            train_size,
            self.config.batch_size,
            self.config.num_train_epochs,
            steps_per_epoch,
            total_train_steps,
            warmup_steps,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
        )

        class_weights = self._compute_class_weights(train_df["label"])
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)

        use_amp = self.device.type == "cuda" and (self.config.fp16 or self.config.bf16)
        scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda" and self.config.fp16)

        best_state_dict = None
        best_val_f1 = -1.0
        best_threshold = 0.5
        epochs_without_improvement = 0
        global_step = 0

        logger.info("[%s | seed=%d] Starting training", run_name, seed)
        train_t0 = time.perf_counter()

        for epoch in range(1, self.config.num_train_epochs + 1):
            model.train()
            epoch_loss_sum = 0.0
            epoch_examples = 0

            for batch_idx, batch in enumerate(train_loader, start=1):
                labels = batch["labels"].to(self.device)
                inputs = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k != "labels"
                }

                optimizer.zero_grad(set_to_none=True)

                if use_amp and self.config.fp16:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        outputs = model(**inputs)
                        logits = outputs.logits
                        loss = loss_fct(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_val)
                    scaler.step(optimizer)
                    scaler.update()
                elif use_amp and self.config.bf16:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = model(**inputs)
                        logits = outputs.logits
                        loss = loss_fct(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_val)
                    optimizer.step()
                else:
                    outputs = model(**inputs)
                    logits = outputs.logits
                    loss = loss_fct(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_val)
                    optimizer.step()

                scheduler.step()

                bs = labels.size(0)
                epoch_loss_sum += loss.item() * bs
                epoch_examples += bs
                global_step += 1

                if global_step % self.config.logging_steps == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    logger.info(
                        "[%s | seed=%d] epoch=%d step=%d/%d loss=%.4f lr=%.6e",
                        run_name,
                        seed,
                        epoch,
                        global_step,
                        total_train_steps,
                        loss.item(),
                        current_lr,
                    )

            train_loss = epoch_loss_sum / max(1, epoch_examples)
            val_metrics = self._evaluate_model(
                model=model,
                dataloader=val_loader,
                threshold=None,
                loss_weights=class_weights,
            )

            logger.info(
                "[%s | seed=%d] epoch=%d train_loss=%.4f val_loss=%.4f val_f1=%.4f best_threshold=%.2f pred_pos_rate=%.4f true_pos_rate=%.4f",
                run_name,
                seed,
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["f1"],
                val_metrics["threshold"],
                val_metrics["positive_rate_pred"],
                val_metrics["positive_rate_true"],
            )

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_threshold = val_metrics["threshold"]
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(
                    "[%s | seed=%d] Early stopping triggered at epoch %d",
                    run_name,
                    seed,
                    epoch,
                )
                break

        logger.info(
            "[%s | seed=%d] Training finished in %.2fs",
            run_name,
            seed,
            time.perf_counter() - train_t0,
        )

        if best_state_dict is None:
            raise RuntimeError("No best model state was captured during training.")

        model.load_state_dict(best_state_dict)

        logger.info(
            "[%s | seed=%d] Evaluating on test set with val-tuned threshold %.2f",
            run_name,
            seed,
            best_threshold,
        )

        test_metrics = self._evaluate_model(
            model=model,
            dataloader=test_loader,
            threshold=best_threshold,
            loss_weights=class_weights,
        )

        logger.info(
            "[%s | seed=%d] Test F1 = %.6f | pred_pos_rate=%.4f | true_pos_rate=%.4f | pred_pos=%d | true_pos=%d",
            run_name,
            seed,
            test_metrics["f1"],
            test_metrics["positive_rate_pred"],
            test_metrics["positive_rate_true"],
            test_metrics["n_pred_positive"],
            test_metrics["n_true_positive"],
        )

        logger.info(
            "[%s | seed=%d] Total run time = %.2fs",
            run_name,
            seed,
            time.perf_counter() - run_t0,
        )

        return float(test_metrics["f1"])

    # --------------------------------------------------
    # Mixed Evaluation
    # --------------------------------------------------

    def evaluate_mixed(self, df: pd.DataFrame, n_runs: int = 1) -> dict[str, float]:
        df = self._prepare_df(df)

        if df["label"].nunique() < 2:
            raise ValueError("Dataset must contain both classes.")

        scores = []
        logger.info("Starting mixed evaluation | rows=%d | n_runs=%d", len(df), n_runs)

        for i in range(n_runs):
            logger.info("Mixed run %d/%d", i + 1, n_runs)

            # target split = 70 / 10 / 20
            train_val_df, test_df = train_test_split(
                df,
                test_size=0.2,
                stratify=df["label"],
                random_state=i,
            )

            if train_val_df["label"].nunique() < 2:
                raise ValueError("train_val split must contain both classes.")

            train_df, val_df = train_test_split(
                train_val_df,
                test_size=0.125,  # 0.125 * 0.8 = 0.1 overall
                stratify=train_val_df["label"],
                random_state=i,
            )

            f1 = self._fit_and_eval(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                seed=i,
                run_name="mixed",
            )
            scores.append(f1)

        result = {
            "f1_mean": float(np.mean(scores)),
            "f1_std": float(np.std(scores)),
        }
        logger.info("Mixed evaluation done: %s", result)
       