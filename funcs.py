import json


def save_to_json(name,dict):
    with open(f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(dict, f, ensure_ascii=False, indent=2)


def get_json(name):
    with open(f"{name}.json", "r", encoding="utf-8") as f:
        return json.load(f)