from dataclasses import dataclass
from xgboost import XGBClassifier


@dataclass
class XGBoostConfig:
    max_depth: int = 8
    learning_rate: float = 0.1
    n_estimators: int = 100
    scale_pos_weight: float = 3.0
    reg_lambda: float = 1.0


def build_xgboost(config: XGBoostConfig = None, random_state: int = 42):
    """
    Returns an XGBClassifier instance.

    Compatible with RepostPredictor.
    """

    config = config or XGBoostConfig()

    return XGBClassifier(
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        scale_pos_weight=config.scale_pos_weight,
        reg_lambda=config.reg_lambda,
        eval_metric="logloss",
        objective="binary:logistic",
        tree_method="hist",
        random_state=random_state,
        enable_categorical=True,
    )