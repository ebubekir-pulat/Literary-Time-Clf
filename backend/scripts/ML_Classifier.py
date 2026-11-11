from pathlib import Path
from pickle import dump, load

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from load_dataset import get_train_and_test_data
from score import print_final_results


# Directory containing this script
SCRIPT_DIR = Path(__file__).resolve().parent


def tfidf_plus_logistic_regress(X_train, X_test, y_train, y_test):
    print("Testing TF-IDF + Logistic Regression")

    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=10000
        )),
        ('classifier', LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced'
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n\nFinal Results of TF-IDF + Logistic Regression")
    print_final_results(y_test, y_pred)

    # Save model
    model_path = SCRIPT_DIR / "tfidf-logreg.pkl"

    with open(model_path, "wb") as f:
        dump(model, f, protocol=5)


def get_trained_tfidf_logreg():
    model_path = SCRIPT_DIR / "tfidf-logreg.pkl"

    with open(model_path, "rb") as f:
        model = load(f)

    return model


def tfidf_charngram_plus_linear_svm(X_train, X_test, y_train, y_test):
    print("\n\nTesting TF-IDF (character n-gram (1, 3)) + Linear SVC")

    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            max_features=10000,
            analyzer='char_wb',
            ngram_range=(1, 3)
        )),
        ('classifier', LinearSVC(
            max_iter=1000
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n\nFinal Results of TF-IDF (character n-gram (1, 3)) + Linear SVC")
    print_final_results(y_test, y_pred)

    # Save model
    model_path = SCRIPT_DIR / "tfidf-linearsvm.pkl"

    with open(model_path, "wb") as f:
        dump(model, f, protocol=5)


def get_trained_tfidf_linearsvm():
    model_path = SCRIPT_DIR / "tfidf-linearsvm.pkl"

    with open(model_path, "rb") as f:
        model = load(f)

    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_and_test_data()

    tfidf_plus_logistic_regress(
        X_train, X_test, y_train, y_test
    )

    tfidf_charngram_plus_linear_svm(
        X_train, X_test, y_train, y_test
    )
