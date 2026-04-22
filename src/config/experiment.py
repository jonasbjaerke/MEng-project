from dataclasses import dataclass


@dataclass
class ExperimentConfig:

    model: str = "xgb"                 # "XGB" | "bert"
    evaluation_mode: str = "all"      # "mixed" | "ood" | "id" | "all"
    dataset_file: str = "dataset.csv"
    save_feature_gains: bool = False


XGB_USER_1TO1 = ExperimentConfig(
    model="xgb",
    evaluation_mode="all",
    dataset_file="dataset_user_1to1.csv",
    save_feature_gains=False,
)

XGB_USER_1TO5 = ExperimentConfig(
    model="xgb",
    evaluation_mode="all",
    dataset_file="dataset_user_1to5.csv",
    save_feature_gains=False,
)

XGB_HYBRID_1TO1 = ExperimentConfig(
    model="xgb",
    evaluation_mode="all",
    dataset_file="dataset_hybrid_1to1.csv",
    save_feature_gains=False,
)

XGB_HYBRID_1TO5 = ExperimentConfig(
    model="xgb",
    evaluation_mode="all",
    dataset_file="dataset_hybrid_1to5.csv",
    save_feature_gains=True,
)

XGB_MESSAGE_1TO1 = ExperimentConfig(
    model="xgb",
    evaluation_mode="all",
    dataset_file="dataset_message_1to1.csv",
    save_feature_gains=False,
)

XGB_MESSAGE_1TO5 = ExperimentConfig(
    model="xgb",
    evaluation_mode="all",
    dataset_file="dataset_message_1to5.csv",
    save_feature_gains=False,
)

BERT_1TO1 = ExperimentConfig(
    model="bert",
    evaluation_mode="mixed",
    dataset_file="dataset_bert_1to1.csv",
    save_feature_gains=False,
)


BERT_1TO5 = ExperimentConfig(
    model="bert",
    evaluation_mode="mixed",
    dataset_file="dataset_bert_1to5.csv",
    save_feature_gains=False,
)

MESSAGE_1TO5_UNIQUE_PID = ExperimentConfig(
    model="xgb",
    evaluation_mode="all",
    dataset_file="dataset_message_1to5_unique_pid.csv",
    save_feature_gains=False,
)


EXPERIMENT_CONFIGS = {
    "XGB_USER_1TO1": XGB_USER_1TO1,
    "XGB_USER_1TO5": XGB_USER_1TO5,
    "XGB_HYBRID_1TO1": XGB_HYBRID_1TO1,
    "XGB_HYBRID_1TO5": XGB_HYBRID_1TO5,
    "XGB_MESSAGE_1TO1": XGB_MESSAGE_1TO1,
    "XGB_MESSAGE_1TO5": XGB_MESSAGE_1TO5,
    "BERT_1TO1": BERT_1TO1,
    "BERT_1TO5": BERT_1TO5,
    "MESSAGE_1TO5_UNIQUE_PID": MESSAGE_1TO5_UNIQUE_PID,
}