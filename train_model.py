import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "labeled_code_dataset.csv")
ARTIFACT_PATH = os.path.join(BASE_DIR, "artifacts", "defect_classifier.joblib")


def main() -> None:
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH).dropna(subset=["code", "label"])
    df = df.drop_duplicates(subset=["code"]).reset_index(drop=True)
    total_samples = len(df)
    print(f"Total deduplicated samples: {total_samples}")

    X = df["code"].astype(str)
    y_binary = df["label"].astype(int)
    y_category = df["category"].astype(str) if "category" in df.columns else None

    # Step 1: Train / Validation / Test separation (70% Train, 15% Val, 15% Test)
    # First split off 15% Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_binary, test_size=0.15, random_state=42, stratify=y_binary
    )
    # Then split Train (70/85 = ~82.35%) and Val (15/85 = ~17.65%)
    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=42, stratify=y_train_val
    )

    print(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Step 2: Feature Union (Word n-grams + Char n-grams)
    features = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b|[^\w\s]",
            max_features=15000,
            sublinear_tf=True
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=25000,
            sublinear_tf=True
        )),
    ])

    # Step 3: Binary Defect Risk Pipeline
    binary_pipeline = Pipeline([
        ("features", features),
        ("clf", LogisticRegression(class_weight="balanced", C=1.5, max_iter=2000, random_state=42)),
    ])

    # Step 4: 5-Fold Stratified Cross-Validation on Training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(binary_pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"5-Fold CV ROC-AUC: Mean={cv_scores.mean():.4f}, Std={cv_scores.std():.4f}")

    # Fit on Train
    binary_pipeline.fit(X_train, y_train)

    # Validation evaluation
    val_preds = binary_pipeline.predict(X_val)
    val_probs = binary_pipeline.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, val_preds)
    val_auc = roc_auc_score(y_val, val_probs)
    val_f1 = f1_score(y_val, val_preds, average="weighted")
    print(f"Validation - Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}")

    # Final fit on Train + Val, then evaluate on Test
    binary_pipeline.fit(X_train_val, y_train_val)
    test_preds = binary_pipeline.predict(X_test)
    test_probs = binary_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds, average="weighted", zero_division=0)
    recall = recall_score(y_test, test_preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)
    auc = roc_auc_score(y_test, test_probs)

    print("\n--- Test Set Evaluation ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    # Step 5: Multi-class Category Pipeline
    cat_pipeline = None
    cat_report = {}
    if y_category is not None:
        X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
            X, y_category, test_size=0.2, random_state=42, stratify=y_category
        )
        cat_features = FeatureUnion([
            ("word_tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b|[^\w\s]", max_features=12000)),
            ("char_tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=20000)),
        ])
        cat_pipeline = Pipeline([
            ("features", cat_features),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        ])
        cat_pipeline.fit(X_tr_c, y_tr_c)
        cat_preds = cat_pipeline.predict(X_te_c)
        cat_report = classification_report(y_te_c, cat_preds, output_dict=True, zero_division=0)
        print("\n--- Multi-Class Defect Category Classification ---")
        print(classification_report(y_te_c, cat_preds, zero_division=0))

    # Save to artifact
    os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
    payload = {
        "pipeline": binary_pipeline,
        "category_pipeline": cat_pipeline,
        "categories": sorted(list(df["category"].unique())) if "category" in df.columns else ["clean", "defect"],
        "metrics": {
            "dataset_size": int(total_samples),
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
            "test_size": int(len(X_test)),
            "test_accuracy": round(float(acc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
            "cv_roc_auc_std": round(float(cv_scores.std()), 4),
            "vectorizer": "TF-IDF Word (1,2) + Char (3,5) Hybrid",
            "classifier": "LogisticRegression(class_weight='balanced')",
            "category_metrics": cat_report,
        },
    }
    joblib.dump(payload, ARTIFACT_PATH)
    print(f"\nSaved updated model and metrics to: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
