import json
from pathlib import Path


def load_json(file: Path) -> dict:
    """Helper function to load json file into dictionary.

    Args:
        file: file path to transcript json data for questions and answers.
    """
    with open(file) as f:
        json_data = json.load(f)
    return json_data


def save_json(file: Path, data: dict) -> None:
    """Helper function to save json file from dictionary.

    Args:
        file: file path to transcript json data for questions and answers.
        json_data: dictionary transcript data to process.
    """
    formatted_dict = json.dumps(data)
    formatted_dict = json.loads(formatted_dict)
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(formatted_dict, f, ensure_ascii=False, indent=4)