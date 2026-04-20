from dataclasses import dataclass


@dataclass
class XGBoostConfig:
    max_depth: int = 8
    learning_rate: float = 0.1
    n_estimators: int = 100
    scale_pos_weight: float = 3.0
    reg_lambda: float = 1.0


@dataclass
class BertConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 128

    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 64
    num_train_epochs: int = 10
    early_stopping_patience: int = 2
    dropout_rate: float = 0.1

    # imbalance handling
    use_class_weights: bool = True

    # device / precision
    force_cpu: bool = False
    fp16: bool = False
    bf16: bool = False

    # speed / experimentation
    logging_steps: int = 100
    sample_size: int | None = None
    num_workers: int = 0

    gradient_clip_val: float = 1.0