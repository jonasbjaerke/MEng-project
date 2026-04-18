from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    @property
    def posts_dir(self) -> Path:
        return self.raw_dir / "posts"

    @property
    def users_dir(self) -> Path:
        return self.raw_dir / "users"

    @property
    def datasets_dir(self) -> Path:
        return self.processed_dir / "datasets"

    @property
    def feature_analysis_dir(self) -> Path:
        return self.results_dir / "xgb" / "feature_analysis"
