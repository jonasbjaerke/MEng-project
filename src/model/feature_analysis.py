import pandas as pd
import matplotlib.pyplot as plt

from ..config.dataset import HYBRID_1TO5
from ..config.paths import PathsConfig
from .xgb_repost_predictor import RepostPredictor
from .xgboost import build_xgboost


def extract_mean_f1(result):
    if "f1_mean" in result:
        return float(result["f1_mean"])

    f1s = [float(v["f1_mean"]) for v in result.values()]
    return sum(f1s) / len(f1s)


def get_next_removable_feature(gain_df, ignored_features):
    """
    Return the highest-gain feature that is not already ignored.
    """
    for _, row in gain_df.iterrows():
        feature = str(row["feature"])
        if feature in ignored_features:
            continue
        return feature
    return None


def main():
    paths_cfg = PathsConfig()
    dataset_cfg = HYBRID_1TO5

    data_path = paths_cfg.datasets_dir / dataset_cfg.output_filename
    output_dir = paths_cfg.feature_analysis_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    results = []
    ignored_features = []
    n_runs = 50

    for run_idx in range(n_runs):
        predictor = RepostPredictor(build_xgboost)

        if ignored_features:
            predictor.ignore_features(ignored_features)

        print(f"\n=== Run {run_idx + 1} / {n_runs} ===")
        print(f"Dataset: {dataset_cfg.output_filename}")
        print(f"Ignored features: {ignored_features}")

        mixed_result = predictor.evaluate_mixed(df)
        ood_result = predictor.evaluate_out_of_distribution(df)

        mixed_f1 = extract_mean_f1(mixed_result)
        ood_avg_f1 = extract_mean_f1(ood_result)

        gain_df = predictor.get_feature_gains()
        top_feature = get_next_removable_feature(gain_df, ignored_features)

        results.append({
            "run": run_idx + 1,
            "n_removed": len(ignored_features),
            "removed_so_far": ", ".join(ignored_features) if ignored_features else "None",
            "mixed_f1": mixed_f1,
            "ood_avg_f1": ood_avg_f1,
            "top_feature_removed_next": top_feature,
        })

        print(f"Mixed F1: {mixed_f1:.4f}")
        print(f"OOD Avg F1: {ood_avg_f1:.4f}")
        print(f"Top removable feature next: {top_feature}")

        if top_feature is None:
            break

        ignored_features.append(top_feature)

    results_df = pd.DataFrame(results)

    plt.figure(figsize=(10, 5))
    plt.plot(results_df["n_removed"], results_df["mixed_f1"], marker="o", label="Mixed F1")
    plt.plot(results_df["n_removed"], results_df["ood_avg_f1"], marker="o", label="OOD Avg F1")

    plt.xlabel("Number of Top Features Removed")
    plt.ylabel("F1 Score")
    plt.title("Effect of Removing Top Features")

    x_vals = results_df["n_removed"].tolist()
    if len(x_vals) > 10:
        step = max(1, len(x_vals) // 10)
        xticks = x_vals[::step]
        if x_vals[-1] not in xticks:
            xticks.append(x_vals[-1])
        plt.xticks(xticks)
    else:
        plt.xticks(x_vals)

    plt.legend()
    plt.tight_layout()

    plot_path = output_dir / "feature_removal_f1_hybrid_1to5.png"
    results_path = output_dir / "feature_removal_f1_hybrid_1to5.csv"

    plt.savefig(plot_path, dpi=300)
    plt.close()

    results_df.to_csv(results_path, index=False)

    print("\nSaved:")
    print(f"- {plot_path}")
    print(f"- {results_path}")
    print("\nResults:")
    print(results_df)


if __name__ == "__main__":
    main()