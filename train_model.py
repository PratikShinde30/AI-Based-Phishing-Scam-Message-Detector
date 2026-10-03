
import pandas as pd
import joblib
import json

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Project directory
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "spam.csv"
MODEL_PATH = BASE_DIR / "spam_detector.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"


def main():
    # 1. Load the dataset
    print("Loading dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place spam.csv in the project folder."
        )

    # Kaggle SMS Spam Collection dataset uses v1 and v2
    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    # 2. Validate and prepare columns
    df.columns = df.columns.str.strip()

    if not {"v1", "v2"}.issubset(df.columns):
        raise ValueError(
            "Expected columns 'v1' and 'v2' were not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[["v1", "v2"]].copy()
    df.columns = ["label", "message"]

    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()

    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"] != ""]
    df = df.drop_duplicates(subset=["message"])

    if df["label"].nunique() != 2:
        raise ValueError("The dataset must contain both ham and spam labels.")

    print("\nDataset information:")
    print(f"Total messages: {len(df)}")
    print("\nClass distribution:")
    print(df["label"].value_counts())

    # 3. Separate input messages and target labels
    X = df["message"]
    y = df["label"].map({"ham": 0, "spam": 1})

    # 4. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 5. Create the machine learning pipeline
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                max_features=20000,
                sublinear_tf=True
            )
        ),
        ("classifier", MultinomialNB(alpha=0.5))
    ])

    # 6. Train the model
    print("\nTraining the model...")
    model.fit(X_train, y_train)

    # 7. Evaluate on unseen test data
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print("\n========== MODEL EVALUATION ==========")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Spam"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    # 8. Save the complete pipeline
    joblib.dump(model, MODEL_PATH)

    # Save evaluation metrics for the Streamlit application
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "total_samples": int(len(df))
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()