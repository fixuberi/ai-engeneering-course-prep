"""Train an Iris classifier.

Run with:  uv run python train.py
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data():
    """Load Iris and split it into stratified train / test sets."""
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    feature_names = list(iris.feature_names)
    target_names = [str(name) for name in iris.target_names]
    return X_train, X_test, y_train, y_test, feature_names, target_names


def describe_split(y_train, y_test, target_names):
    """Print how many samples of each class landed in train and test."""
    for split_name, y in (("train", y_train), ("test", y_test)):
        counts = np.bincount(y, minlength=len(target_names))
        per_class = ", ".join(f"{name}={n}" for name, n in zip(target_names, counts))
        print(f"  {split_name:<5} {len(y):>3} samples: {per_class}")


def build_pipeline():
    """Scale features, then classify with logistic regression."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def main():
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()

    print("Features:", feature_names)
    print("Classes: ", target_names)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print("Class balance:")
    describe_split(y_train, y_test, target_names)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=3))


if __name__ == "__main__":
    main()
