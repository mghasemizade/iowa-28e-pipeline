"""
Stage A.5 — Classification: Embedding-Based

Extracts contract embeddings from GPT text-embedding-3-small, a local LLaMA
3.1 70B, and Gemini gemini-embedding-001, and trains downstream classifiers
(logistic regression, SVM, ridge, extra trees, MLP) to predict institutional
form. Hyperparameter search space and CV config ported verbatim from paper
Appendix Listing 1.

Uses the same data/labeled/train.csv + test.csv schema as
chain_of_thought.py (PDF_ID, Folder_Name, surya_ocr, Purpose, gpt_summary,
gemini_summary, llama_summary, Institutional_Form — renamed on load to
contract_id/purpose/label) for labels, and precomputed embeddings from
data/labeled/embeddings/{split}_{model}.npy for features — one row per
contract, in the same order as the corresponding CSV (train.csv: N = 901,
test.csv: N = 227). All three embedding sources share the same recipe —
"Purpose: ... \nContract Summary: {model's own summary column} \nFolder
Name:..." (gpt_summary / gemini_summary / llama_summary respectively) — only
the embedding call itself (API vs. local HF model, chunking) differs per
provider; see generate_embeddings.py for the exact recipe each was produced
with.
Hyperparameters are selected via 5-fold stratified CV on the train split
only; the best estimator per embedding source is then evaluated once on the
held-out test split (N = 227), matching paper Tables 3-5.

Results reported in paper Tables 3-5 (GPT / LLaMA / Gemini best classifiers).

Note on --apply-full-corpus: that pass classifies every contract in the
non-public full 21,629-agreement corpus (data/ocr_output/summaries/), which
is not part of the public release — it requires your own raw PDFs and a
completed run of Stages A.2/A.3 (see those scripts' docstrings), and (unlike
the train/CV/test path) calls the live embedding APIs via embed_text since
no precomputed .npy exists for that corpus.



Input:  data/labeled/train.csv, data/labeled/test.csv (ground truth labels)
        + data/labeled/embeddings/{train,test}_{gpt,llama,gemini}.npy
        (precomputed embeddings, row-aligned with the CSVs) — the only
        inputs needed for the train/CV/test path, all part of the public
        release; --apply-full-corpus additionally needs the non-public
        data/ocr_output/summaries/{contract_id}_{model}.json
Output: data/labeled/embedding_predictions_{model}.csv,
        data/labeled/embedding_classification_report_{model}.json,
        data/labeled/full_corpus_classifications_{model}.csv (--apply-full-corpus
            only; columns contract_id, predicted_label, prediction_confidence,
            model — prediction_confidence is what Stage A.6 filters on)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from src.common.embeddings import embed_text

SUMMARY_DIR = Path("data/ocr_output/summaries")
LABELED_DIR = Path("data/labeled")
TRAIN_PATH = LABELED_DIR / "train.csv"
TEST_PATH = LABELED_DIR / "test.csv"
EMBEDDINGS_DIR = LABELED_DIR / "embeddings"

EMBEDDING_MODELS = ["gpt", "llama", "gemini"]
CATEGORIES = ["Service Contract", "Resource Sharing", "Joint Operations", "New Joint Entities"]

_COLUMN_RENAME = {"PDF_ID": "contract_id", "Purpose": "purpose", "Institutional_Form": "label"}


def _load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"PDF_ID": str}).rename(columns=_COLUMN_RENAME)
    # Institutional_Form ships lowercase ("service contract"); CATEGORIES
    # (and classification_report's labels=CATEGORIES) is title-case.
    df["label"] = df["label"].str.strip().str.title()
    return df


def _model_search_space(seed: int | None):
    """(name, estimator, param_grid) tuples — ported verbatim from paper
    Appendix Listing 1."""
    return [
        (
            "logreg",
            Pipeline([
                ("scale", StandardScaler()),
                ("pca", "passthrough"),
                ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
            ]),
            {"pca": ["passthrough", PCA(n_components=0.95)], "clf__C": [0.1, 1, 10]},
        ),
        (
            "linear_svm",
            Pipeline([
                ("scale", StandardScaler()),
                ("pca", "passthrough"),
                ("clf", LinearSVC(class_weight="balanced")),
            ]),
            {"pca": ["passthrough", PCA(n_components=0.95)], "clf__C": [0.1, 1, 10]},
        ),
        (
            "rbf_svm",
            Pipeline([
                ("scale", StandardScaler()),
                ("pca", "passthrough"),
                ("clf", SVC(kernel="rbf", class_weight="balanced")),
            ]),
            {
                "pca": ["passthrough", PCA(n_components=0.95)],
                "clf__C": [1, 10, 100],
                "clf__gamma": ["scale", 1e-2, 1e-1],
            },
        ),
        (
            "ridge",
            Pipeline([
                ("scale", StandardScaler()),
                ("pca", "passthrough"),
                ("clf", RidgeClassifier(class_weight="balanced")),
            ]),
            {"pca": ["passthrough", PCA(n_components=0.95)], "clf__alpha": [0.1, 1, 10]},
        ),
        (
            "trees",
            ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, class_weight="balanced", random_state=seed),
            {"max_depth": [None, 20], "min_samples_leaf": [1, 2], "max_features": ["sqrt", 0.5]},
        ),
        (
            "mlp",
            Pipeline([
                ("scale", StandardScaler()),
                ("pca", "passthrough"),
                ("clf", MLPClassifier(max_iter=2000, early_stopping=True, random_state=seed)),
            ]),
            {
                "pca": ["passthrough", PCA(n_components=0.95)],
                "clf__hidden_layer_sizes": [(128,), (256,)],
                "clf__alpha": [1e-4, 1e-3],
            },
        ),
    ]


def extract_embedding(summary_text: str, model_key: str) -> np.ndarray:
    return embed_text(summary_text, model_key)


def _load_split_embeddings(split_name: str, model_key: str, n_rows: int) -> np.ndarray:
    """Load the precomputed embeddings for split_name ("train"/"test") and
    model_key from data/labeled/embeddings/{split_name}_{model_key}.npy.
    Rows are assumed to be in the same order as the corresponding CSV."""
    path = EMBEDDINGS_DIR / f"{split_name}_{model_key}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing precomputed embeddings for {model_key}: {path}. Expected one row "
            f"per contract, in the same order as {split_name}.csv."
        )
    X = np.load(path)
    if X.shape[0] != n_rows:
        raise ValueError(
            f"{path} has {X.shape[0]} rows but {split_name}.csv has {n_rows} — "
            "embeddings must be row-aligned with the CSV."
        )
    return X


def run_for_model(model_key: str, train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int | None):
    train_ids, y_train = train_df["contract_id"].tolist(), train_df["label"].to_numpy()
    test_ids, y_test = test_df["contract_id"].tolist(), test_df["label"].to_numpy()
    X_train = _load_split_embeddings("train", model_key, len(train_df))
    X_test = _load_split_embeddings("test", model_key, len(test_df))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    best_name, best_estimator, best_score = None, None, -np.inf
    for name, estimator, param_grid in _model_search_space(seed):
        search = GridSearchCV(estimator, param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1)
        search.fit(X_train, y_train)

        print(f"{model_key}/{name}: best CV f1_weighted = {search.best_score_:.3f}")
        if search.best_score_ > best_score:
            best_name, best_estimator, best_score = name, search.best_estimator_, search.best_score_

    if best_estimator is None:
        print(f"[WARN] every candidate classifier failed to fit for {model_key}, skipping")
        return None

    y_pred = best_estimator.predict(X_test)
    report = classification_report(y_test, y_pred, labels=CATEGORIES, output_dict=True, zero_division=0)
    report["best_classifier"] = best_name
    report["best_cv_f1_weighted"] = best_score
    print(f"{model_key} best={best_name} test weighted F1: {report['weighted avg']['f1-score']:.3f}")

    pd.DataFrame({"contract_id": test_ids, "true_label": y_test, "predicted_label": y_pred}).to_csv(
        LABELED_DIR / f"embedding_predictions_{model_key}.csv", index=False
    )
    (LABELED_DIR / f"embedding_classification_report_{model_key}.json").write_text(json.dumps(report, indent=2))
    return best_estimator


def _predict_with_confidence(estimator, X: np.ndarray):
    """Predict labels plus a per-row confidence score in [0, 1].

    Which of the six candidate classifiers wins CV varies by embedding
    source, and only some of them (logreg, trees, mlp) expose predict_proba
    — linear_svm/rbf_svm/ridge only expose decision_function. Softmax the
    decision scores into a proba-like confidence for those so downstream
    (Stage A.6's prediction_confidence >= threshold filter) works regardless
    of which estimator was selected."""
    predictions = estimator.predict(X)
    if hasattr(estimator, "predict_proba"):
        confidence = estimator.predict_proba(X).max(axis=1)
    else:
        scores = np.atleast_2d(estimator.decision_function(X))
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        confidence = (exp_scores / exp_scores.sum(axis=1, keepdims=True)).max(axis=1)
    return predictions, confidence


def classify_full_corpus(model_key: str, fitted_estimator, out_path: Path):
    """Apply the fitted classifier to every contract summary available for
    model_key (not just the labeled set) — this is what Section 5 refers to
    as "we apply the classification model described in Section 3" across
    the full 21,629-agreement corpus, and is what Stage A.6 (financial
    extraction) filters down to Service Contract rows from.

    Requires the non-public data/ocr_output/summaries/*_{model}.json corpus
    (see Stage A.2/A.3 docstrings) — not reproducible from the published
    train.csv/test.csv alone."""
    summary_paths = sorted(SUMMARY_DIR.glob(f"*_{model_key}.json"))
    if not summary_paths:
        print(f"[WARN] no summaries found for {model_key}, skipping full-corpus classification")
        return

    contract_ids, vectors = [], []
    for path in summary_paths:
        record = json.loads(path.read_text())
        try:
            vectors.append(extract_embedding(record["summary"], model_key))
            contract_ids.append(record["contract_id"])
        except Exception as exc:  # noqa: BLE001 — log and continue the batch
            print(f"[ERROR] embedding failed for {record.get('contract_id')} ({model_key}): {exc}")

    predictions, confidence = _predict_with_confidence(fitted_estimator, np.stack(vectors))
    pd.DataFrame({
        "contract_id": contract_ids,
        "predicted_label": predictions,
        "prediction_confidence": confidence,
        "model": model_key,
    }).to_csv(out_path, index=False)
    print(f"Wrote full-corpus classifications for {model_key} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run embedding-based classification.")
    parser.add_argument("--model", choices=EMBEDDING_MODELS + ["all"], default="all")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument(
        "--apply-full-corpus", action="store_true",
        help="After fitting, classify every summary in data/ocr_output/summaries/ "
             "(not just the labeled set) and write full_corpus_classifications_{model}.csv "
             "— this is the input Stage A.6 filters to Service Contracts. Requires the "
             "non-public full OCR corpus (your own PDFs run through Stages A.2/A.3); "
             "not usable from the public release alone.",
    )
    args = parser.parse_args()

    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    train_df = _load_split(args.train)
    test_df = _load_split(args.test)

    for model_key in EMBEDDING_MODELS if args.model == "all" else [args.model]:
        best_estimator = run_for_model(model_key, train_df, test_df, args.seed)
        if args.apply_full_corpus and best_estimator is not None:
            classify_full_corpus(
                model_key, best_estimator, LABELED_DIR / f"full_corpus_classifications_{model_key}.csv"
            )


if __name__ == "__main__":
    main()
