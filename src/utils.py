
import json
from pathlib import Path
from datetime import datetime, timezone
import logging

def write_json(data, path):
    """
    Write dict to JSON file.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_json(path: str | Path):
    """
    Load JSON file and return parsed object.
    Accepts filename or full path.
    """

    path = Path(path)

    if path.suffix != ".json":
        path = path.with_suffix(".json")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def parse_dt(ts):
        if not ts:
            return None
        try:
            return datetime.fromisoformat(
                ts.replace("Z", "+00:00")
            ).astimezone(timezone.utc)
        except Exception:
            return None
    


def get_logger(name: str = __name__):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger(name)
        


def save_csv(df, path):
    """
    Save DataFrame to CSV.
    Ensures parent directory exists.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

    return path

