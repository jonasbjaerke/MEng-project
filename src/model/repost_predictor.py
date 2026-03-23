import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score


class RepostPredictor:
    """
    Generic repost prediction framework.

    Supports:
        - Mixed split evaluation
        - In-distribution evaluation
        - Out-of-distribution evaluation
    """

    def __init__(self, model_builder):
        """
        model_builder: function(random_state=...) -> sklearn model
        id_cols: columns to drop before training
        """
        
        self.model_builder = model_builder
        self.id_cols = ["A_id", "S_id", "P_id"]

        self._feature_gains = None
        self._feature_names = None

    # --------------------------------------------------
    # Data Preparation
    # --------------------------------------------------

    def _prepare(self, df):
        X = df.drop(columns=self.id_cols + ["label","U-P_R_FollowS","U-HA_S_RetweetedRate","U-HA-R_repostsS","U-HA_S_LikedRate"]).copy()
        y = df["label"].copy()

        # Handle categorical columns
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            X[col] = X[col].fillna("missing").astype(str).astype("category")

        # Handle numeric columns
        num_cols = X.select_dtypes(include=["number", "bool"]).columns
        for col in num_cols:
            X[col] = X[col].fillna(0)

        return X, y

    # --------------------------------------------------
    # Mixed Evaluation
    # --------------------------------------------------

    def evaluate_mixed(self, df, n_runs=3):
        X, y = self._prepare(df)
        scores = []
        gains = []

        for i in range(n_runs):
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                stratify=y,
                random_state=i
            )

            model = self.model_builder(random_state=i)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            scores.append(f1_score(y_test, y_pred))

        if hasattr(model, "feature_importances_"):
            gains.append(model.feature_importances_)

        # store gains
        if gains:
            self._feature_gains = np.mean(gains, axis=0)
            self._feature_names = list(X.columns)

        return {
            "f1_mean": np.mean(scores),
            "f1_std": np.std(scores)
        }

    # --------------------------------------------------
    # In-Distribution (per hashtag)
    # --------------------------------------------------

    def evaluate_in_distribution(self, df, n_runs=3):
        results = {}

        for tag in df["hashtag"].unique():
            df_tag = df[df["hashtag"] == tag]
            results[tag] = self.evaluate_mixed(df_tag, n_runs)

        return results

    # --------------------------------------------------
    # Out-of-Distribution (Leave-One-Hashtag-Out)
    # --------------------------------------------------

    def evaluate_out_of_distribution(self, df, n_splits=3):
        results = {}

        for tag in df["hashtag"].unique():

            train_df = df[df["hashtag"] != tag]
            test_df = df[df["hashtag"] == tag]

            X_train_full, y_train_full = self._prepare(train_df)
            X_test, y_test = self._prepare(test_df)

            # --------------------------------------------------
            # Align categorical columns (critical for OOD)
            # --------------------------------------------------
            cat_cols = X_train_full.select_dtypes(include="category").columns

            for col in cat_cols:
                train_categories = X_train_full[col].cat.categories

                # Replace unseen categories in test with "missing"
                X_test[col] = X_test[col].astype(str)
                X_test[col] = X_test[col].where(
                    X_test[col].isin(train_categories),
                    "missing"
                )

                # Ensure category dtype and same category set
                X_test[col] = X_test[col].astype("category")
                X_test[col] = X_test[col].cat.set_categories(train_categories)

            # --------------------------------------------------
            # K-Fold training on training data
            # --------------------------------------------------
            scores = []
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            for train_index, _ in kf.split(X_train_full):
                X_train = X_train_full.iloc[train_index]
                y_train = y_train_full.iloc[train_index]

                model = self.model_builder(random_state=42)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                scores.append(f1_score(y_test, y_pred))

            results[tag] = {
                "f1_mean": np.mean(scores),
                "f1_std": np.std(scores)
            }

        return results
    
    def get_feature_gains(self):

        if self._feature_gains is None:
            raise ValueError("Run evaluate_mixed() first to compute gains.")

        gain_df = pd.DataFrame({
            "feature": self._feature_names,
            "gain": self._feature_gains
        })

        gain_df = gain_df.sort_values("gain", ascending=False)

        return gain_df