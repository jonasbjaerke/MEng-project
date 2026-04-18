from xgboost import XGBClassifier

from ..config.model import XGBoostConfig


def build_xgboost(config: XGBoostConfig | None = None, random_state: int = 42):
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