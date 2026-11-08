import json
from sklearn.model_selection import train_test_split
from prepare_dataset import write_dataset

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


def classify_year_to_period(year):
    if year >= 1800 and year < 1833:
        return "1800-1832"
    
    elif year >= 1833 and year < 1867:
        return "1833-1866"

    elif year >= 1867 and year < 1900:
        return "1867-1899"


def get_train_and_test_data():
    datasets = get_dataset()

    X_train, X_test = [], []
    y_train, y_test = [], []

    for dataset in datasets:
        X = [sample["text"] for sample in dataset]
        y = [classify_year_to_period(sample["year"]) for sample in dataset]

        X_train_part, X_test_part, y_train_part, y_test_part = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        X_train.extend(X_train_part)
        X_test.extend(X_test_part)
        y_train.extend(y_train_part)
        y_test.extend(y_test_part)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    get_dataset()