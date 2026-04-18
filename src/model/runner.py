import argparse
import json
from datetime import datetime

import pandas as pd

from ..config.experiment import EXPERIMENT_CONFIGS
from ..config.paths import PathsConfig
from ..config.model import BertConfig
from .xgb_repost_predictor import RepostPredictor
from .bert_repost_predictor import BertRepostPredictor
from .xgboost import build_xgboost



def main():
    parser = argparse.ArgumentParser(description="Run model experiments")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help="Named experiment config to use",
    )
    parser.add_argument(
        "--save",
        type=str,
        choices=["y", "n"],
        default="n",
        help="Whether to save results (y/n). Default: n",
    )

    args = parser.parse_args()

    exp_cfg = EXPERIMENT_CONFIGS[args.config]
    paths_cfg = PathsConfig()

    data_path = paths_cfg.datasets_dir / exp_cfg.dataset_file
    df = pd.read_csv(data_path)

    if exp_cfg.model == "xgb":
        predictor = RepostPredictor(build_xgboost)

    elif exp_cfg.model == "bert":
        predictor = BertRepostPredictor(BertConfig())

    else:
        raise ValueError(f"Unsupported model type: {exp_cfg.model}")

    print(f"\nRunning config: {args.config}")
    print(f"Running model: {exp_cfg.model}")
    print(f"Running file: {exp_cfg.dataset_file}")
    print(f"Save outputs: {args.save}")

    results = {}

    if exp_cfg.evaluation_mode in ["mixed", "all"]:
        print("\nMixed:")
        mixed_result = predictor.evaluate_mixed(df)
        print(mixed_result)
        results["mixed"] = mixed_result

    if exp_cfg.evaluation_mode in ["ood", "all"]:
        print("\nOut-of-Distribution:")
        ood_result = predictor.evaluate_out_of_distribution(df)
        print(ood_result)
        results["ood"] = ood_result

    if exp_cfg.evaluation_mode in ["id", "all"]:
        print("\nIn-Distribution:")
        id_result = predictor.evaluate_in_distribution(df)
        print(id_result)
        results["id"] = id_result

    if args.save == "y":
        output_dir = paths_cfg.results_dir / exp_cfg.model
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{args.config}_{timestamp}.json"

        payload = {
            "config_name": args.config,
            "model": exp_cfg.model,
            "dataset_file": exp_cfg.dataset_file,
            "evaluation_mode": exp_cfg.evaluation_mode,
            "results": results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    if args.save == "y" and exp_cfg.model == "xgb" and exp_cfg.save_feature_gains:
        print("\nFeature importance:")
        gains_df = predictor.get_feature_gains()
        print(gains_df)

        output_dir = paths_cfg.feature_analysis_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{data_path.stem}_feature_gains.csv"
        output_path = output_dir / filename

        gains_df.to_csv(output_path, index=False)
        print(f"\nFeature gains saved to: {output_path}")


if __name__ == "__main__":
    main()