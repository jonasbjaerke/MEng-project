import copy
import json
import logging
import math
import os
import random
from ..config.model import BertConfig
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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




class BertRepostPredictor:
    """
    Mixed-hashtag BERT repost predictor.

    """

    def __init__(self, config: BertConfig | None = None):
        self.config = config or BertConfig()
        self.required_cols = ["text", "label", "hashtag"]

        self.device = get_device(force_cpu=self.config.force_cpu)
        logger.info("Using device: %s", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
        )
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def _validate_df(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _subsample_if_needed(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.sample_size is None or len(df) <= self.config.sample_size:
            return df

        sampled_parts = []
        total_rows = len(df)
        for _, group in df.groupby("label"):
            frac = len(group) / total_rows
            n_label = max(1, int(round(self.config.sample_size * frac)))
            n_label = min(n_label, len(group))
            sampled_parts.append(group.sample(n=n_label, random_state=42))

        sampled = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=42)
        return sampled.head(self.config.sample_size).reset_index(drop=True)

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

    def _tokenize_df(self, df: pd.DataFrame, split_name: str) -> Dataset:
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

        ds = ds.map(tokenize, batched=True, desc=f"Tokenizing {split_name}")
        ds = ds.rename_column("label", "labels")

        columns = ["input_ids", "attention_mask", "labels"]
        if "token_type_ids" in ds.column_names:
            columns = ["input_ids", "attention_mask", "token_type_ids", "labels"]

        ds.set_format(type="torch", columns=columns)
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

    def _make_model(self) -> AutoModelForSequenceClassification:
        hf_config = AutoConfig.from_pretrained(
            self.config.model_name,
            num_labels=2,
            hidden_dropout_prob=self.config.dropout_rate,
            attention_probs_dropout_prob=self.config.dropout_rate,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            config=hf_config,
        )
        return model.to(self.device)

    def _get_class_weights(self, train_df: pd.DataFrame) -> torch.Tensor | None:
        if not self.config.use_class_weights:
            return None

        class_counts = train_df["label"].value_counts().sort_index()
        n0 = int(class_counts.get(0, 0))
        n1 = int(class_counts.get(1, 0))

        if n0 == 0 or n1 == 0:
            raise ValueError("Both classes must be present to compute class weights.")

        # inverse-frequency style weights, normalized around 1.0
        total = n0 + n1
        w0 = total / (2.0 * n0)
        w1 = total / (2.0 * n1)

        weights = torch.tensor([w0, w1], dtype=torch.float32, device=self.device)
        logger.info(
            "Using class weights: class_0=%.4f class_1=%.4f | counts: n0=%d n1=%d",
            w0,
            w1,
            n0,
            n1,
        )
        return weights

    def _forward_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        class_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = model(**inputs)
        logits = outputs.logits

        if class_weights is None:
            loss = F.cross_entropy(logits, labels)
        else:
            loss = F.cross_entropy(logits, labels, weight=class_weights)

        return loss, logits

    def _evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        class_weights: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        model.eval()

        all_logits = []
        all_labels = []
        total_loss = 0.0
        total_examples = 0

        with torch.no_grad():
            for batch in dataloader:
                labels = batch["labels"].to(self.device)
                inputs = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k != "labels"
                }

                loss, logits = self._forward_loss(
                    model=model,
                    inputs=inputs,
                    labels=labels,
                    class_weights=class_weights,
                )

                bs = labels.size(0)
                total_loss += loss.item() * bs
                total_examples += bs

                all_logits.append(logits.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

        logits_np = np.concatenate(all_logits, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)

        preds = np.argmax(logits_np, axis=1)
        f1 = f1_score(labels_np, preds, zero_division=0)

        return {
            "loss": total_loss / max(1, total_examples),
            "f1": float(f1),
            "labels": labels_np,
            "preds": preds,
            "logits": logits_np,
        }

    def _config_to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.config.model_name,
            "max_length": self.config.max_length,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "batch_size": self.config.batch_size,
            "num_train_epochs": self.config.num_train_epochs,
            "early_stopping_patience": self.config.early_stopping_patience,
            "dropout_rate": self.config.dropout_rate,
            "use_class_weights": self.config.use_class_weights,
            "force_cpu": self.config.force_cpu,
            "fp16": self.config.fp16,
            "bf16": self.config.bf16,
            "logging_steps": self.config.logging_steps,
            "sample_size": self.config.sample_size,
            "num_workers": self.config.num_workers,
            "gradient_clip_val": self.config.gradient_clip_val,
        }

    def _save_experiment_results(
        self,
        result_dir: str | os.PathLike,
        experiment_name: str,
        summary: dict[str, Any],
        run_details: list[dict[str, Any]],
    ) -> None:
        result_path = Path(result_dir)
        result_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{experiment_name}_{timestamp}"

        payload = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "device": str(self.device),
            "config": self._config_to_dict(),
            "summary": summary,
            "runs": run_details,
        }

        json_path = result_path / f"{base_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        runs_df = pd.DataFrame(run_details)
        csv_path = result_path / f"{base_name}_runs.csv"
        runs_df.to_csv(csv_path, index=False)

        summary_df = pd.DataFrame([summary])
        summary_csv_path = result_path / f"{base_name}_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)

        latex_path = result_path / f"{base_name}_latex.txt"
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(f"{summary['f1_mean']:.4f} $\\pm$ {summary['f1_std']:.4f}\n")

        logger.info("Saved results to %s", result_path)
        logger.info("JSON: %s", json_path)
        logger.info("Runs CSV: %s", csv_path)
        logger.info("Summary CSV: %s", summary_csv_path)
        logger.info("LaTeX snippet: %s", latex_path)

    def _fit_and_eval(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        seed: int,
        run_name: str,
    ) -> dict[str, Any]:
        set_global_seed(seed)

        train_df = self._prepare_df(train_df)
        val_df = self._prepare_df(val_df)
        test_df = self._prepare_df(test_df)

        if train_df["label"].nunique() < 2:
            raise ValueError("Training split must contain both classes.")
        if val_df["label"].nunique() < 2:
            raise ValueError("Validation split must contain both classes.")
        if test_df["label"].nunique() < 2:
            raise ValueError("Test split must contain both classes.")

        logger.info(
            "[%s | seed=%d] train=%d val=%d test=%d",
            run_name, seed, len(train_df), len(val_df), len(test_df)
        )

        train_ds = self._tokenize_df(train_df, "train")
        val_ds = self._tokenize_df(val_df, "val")
        test_ds = self._tokenize_df(test_df, "test")

        train_loader = self._make_dataloader(train_ds, train=True)
        val_loader = self._make_dataloader(val_ds, train=False)
        test_loader = self._make_dataloader(test_ds, train=False)

        model = self._make_model()
        class_weights = self._get_class_weights(train_df)

        steps_per_epoch = max(1, math.ceil(len(train_ds) / self.config.batch_size))
        total_train_steps = steps_per_epoch * self.config.num_train_epochs

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=total_train_steps,
        )

        use_amp = self.device.type == "cuda" and (self.config.fp16 or self.config.bf16)
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and self.config.fp16,
        )

        best_state_dict = None
        best_val_loss = float("inf")
        best_val_f1 = float("-inf")
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, self.config.num_train_epochs + 1):
            model.train()
            epoch_loss_sum = 0.0
            epoch_examples = 0

            for batch in train_loader:
                labels = batch["labels"].to(self.device)
                inputs = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k != "labels"
                }

                optimizer.zero_grad(set_to_none=True)

                if use_amp and self.config.fp16:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        loss, _ = self._forward_loss(
                            model=model,
                            inputs=inputs,
                            labels=labels,
                            class_weights=class_weights,
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.gradient_clip_val,
                    )
                    scaler.step(optimizer)
                    scaler.update()

                elif use_amp and self.config.bf16:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        loss, _ = self._forward_loss(
                            model=model,
                            inputs=inputs,
                            labels=labels,
                            class_weights=class_weights,
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.gradient_clip_val,
                    )
                    optimizer.step()

                else:
                    loss, _ = self._forward_loss(
                        model=model,
                        inputs=inputs,
                        labels=labels,
                        class_weights=class_weights,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.gradient_clip_val,
                    )
                    optimizer.step()

                scheduler.step()

                bs = labels.size(0)
                epoch_loss_sum += loss.item() * bs
                epoch_examples += bs

            train_loss = epoch_loss_sum / max(1, epoch_examples)
            val_metrics = self._evaluate_model(
                model=model,
                dataloader=val_loader,
                class_weights=class_weights,
            )

            logger.info(
                "[%s | seed=%d] epoch=%d train_loss=%.4f val_loss=%.4f val_f1=%.4f",
                run_name,
                seed,
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["f1"],
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = float(val_metrics["loss"])
                best_val_f1 = float(val_metrics["f1"])
                best_state_dict = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(
                    "[%s | seed=%d] Early stopping at epoch %d",
                    run_name,
                    seed,
                    epoch,
                )
                break

        if best_state_dict is None:
            raise RuntimeError("No best model state was captured during training.")

        model.load_state_dict(best_state_dict)
        test_metrics = self._evaluate_model(
            model=model,
            dataloader=test_loader,
            class_weights=class_weights,
        )

        logger.info(
            "[%s | seed=%d] Test F1=%.6f Test Loss=%.4f",
            run_name,
            seed,
            test_metrics["f1"],
            test_metrics["loss"],
        )

        return {
            "seed": int(seed),
            "run_name": run_name,
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
            "test_size": int(len(test_df)),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "best_val_f1": float(best_val_f1),
            "test_f1": float(test_metrics["f1"]),
            "test_loss": float(test_metrics["loss"]),
        }

    def evaluate_mixed(
        self,
        df: pd.DataFrame,
        n_runs: int = 2,
        save_results: bool = True,
        result_dir: str = "results",
        experiment_name: str = "bert_mixed",
    ) -> dict[str, float]:
        """
        Mixed setting:
        - Monte Carlo runs
        - 70% train / 30% test
        - validation = 10% of training set
        """
        df = self._prepare_df(df)

        if df["label"].nunique() < 2:
            raise ValueError("Dataset must contain both classes.")

        run_details: list[dict[str, Any]] = []
        logger.info("Starting mixed evaluation | rows=%d | n_runs=%d", len(df), n_runs)

        for i in range(n_runs):
            logger.info("Mixed run %d/%d", i + 1, n_runs)

            train_df, test_df = train_test_split(
                df,
                test_size=0.30,
                stratify=df["label"],
                random_state=i,
            )

            train_df, val_df = train_test_split(
                train_df,
                test_size=0.10,
                stratify=train_df["label"],
                random_state=i,
            )

            run_result = self._fit_and_eval(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                seed=i,
                run_name="mixed",
            )
            run_details.append(run_result)

        f1_scores = [r["test_f1"] for r in run_details]

        summary = {
            "dataset_rows": int(len(df)),
            "n_runs": int(n_runs),
            "f1_mean": float(np.mean(f1_scores)),
            "f1_std": float(np.std(f1_scores)),
        }

        if save_results:
            self._save_experiment_results(
                result_dir=result_dir,
                experiment_name=experiment_name,
                summary=summary,
                run_details=run_details,
            )

        logger.info("Mixed evaluation done: %s", summary)
        return summary