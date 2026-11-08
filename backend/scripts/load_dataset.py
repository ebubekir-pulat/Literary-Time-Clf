import json
from backend.scripts.prepare_dataset import write_dataset

def get_dataset():
    write_dataset()

    # Book exceprts from 1800 to 1832 (inclusive)
    with open("../data/first_third.json", "r", encoding="utf-8") as f:
        first_third = json.load(f)

    # Book exceprts from 1833 to 1866 (inclusive)
    with open("../data/second_third.json", "r", encoding="utf-8") as f:
        second_third = json.load(f)

    # Book exceprts from 1867 to 1899 (inclusive)
    with open("../data/third_third.json", "r", encoding="utf-8") as f:
        third_third = json.load(f)

    return first_third, second_third, third_third


if __name__ == "__main__":
    get_dataset()