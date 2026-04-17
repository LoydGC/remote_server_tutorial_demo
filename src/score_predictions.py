"""
Score all student predictions and produce a leaderboard.

Usage (run as instructor on the server):
    python src/score_predictions.py

Reads:
    /home/lcrowl/remote_server_tutorial_demo/data/test.csv
        Ground-truth test labels (must have columns: ID, Recidivism_Within_3years)

    /home/mlfp_shared/nij_data/predictions/*.csv
        One file per student, named <andrew_id>.csv
        Each file must have columns:
            ID                          - matches test set
            prob_recidivism             - predicted probability (0-1)
            pred_recidivism             - discrete prediction (0 or 1)

Writes:
    /home/mlfp_shared/nij_data/leaderboard.csv   - full results table, sorted by AUC
    /home/mlfp_shared/nij_data/leaderboard.md    - markdown table, easy to open in VS Code
"""

import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

TRUTH_PATH = Path("/home/lcrowl/remote_server_tutorial_demo/data/test.csv")
PRED_DIR   = Path("/home/mlfp_shared/nij_data/predictions")
OUT_DIR    = Path("/home/mlfp_shared/nij_data")

OUTCOME_COL = "Recidivism_Within_3years"


def load_truth() -> pd.DataFrame:
    df = pd.read_csv(TRUTH_PATH)
    if OUTCOME_COL not in df.columns:
        sys.exit(f"ERROR: '{OUTCOME_COL}' column not found in {TRUTH_PATH}")
    df[OUTCOME_COL] = (df[OUTCOME_COL] == "Yes").astype(int) if df[OUTCOME_COL].dtype == object else df[OUTCOME_COL]
    return df[["ID", OUTCOME_COL]].set_index("ID")


def score_one(pred_path: Path, truth: pd.DataFrame) -> dict:
    andrew_id = pred_path.stem
    try:
        preds = pd.read_csv(pred_path)
        required = {"ID", "prob_recidivism", "pred_recidivism"}
        missing = required - set(preds.columns)
        if missing:
            return {"andrew_id": andrew_id, "error": f"missing columns: {missing}"}

        preds = preds.set_index("ID")
        merged = truth.join(preds, how="inner")

        if len(merged) == 0:
            return {"andrew_id": andrew_id, "error": "no matching IDs with truth set"}

        y_true  = merged[OUTCOME_COL]
        y_prob  = merged["prob_recidivism"]
        y_pred  = merged["pred_recidivism"]

        auc       = round(roc_auc_score(y_true, y_prob), 4)
        accuracy  = round(accuracy_score(y_true, y_pred), 4)
        precision = round(precision_score(y_true, y_pred, zero_division=0), 4)
        recall    = round(recall_score(y_true, y_pred, zero_division=0), 4)
        f1        = round(f1_score(y_true, y_pred, zero_division=0), 4)
        n         = len(merged)

        return {
            "andrew_id": andrew_id,
            "auc":       auc,
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "n_scored":  n,
            "error":     "",
        }

    except Exception as e:
        return {"andrew_id": andrew_id, "error": str(e)}


def build_markdown_table(df: pd.DataFrame) -> str:
    lines = []
    # Separate scored and errored rows
    scored = df[df["error"] == ""].copy()
    errored = df[df["error"] != ""].copy()

    header = "| Rank | Andrew ID | AUC | Accuracy | Precision | Recall | F1 | N Scored |"
    sep    = "|------|-----------|-----|----------|-----------|--------|----|----------|"
    lines.append("# NIJ Recidivism Challenge — Leaderboard\n")
    lines.append(header)
    lines.append(sep)
    for rank, (_, row) in enumerate(scored.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['andrew_id']} | {row['auc']} | {row['accuracy']} "
            f"| {row['precision']} | {row['recall']} | {row['f1']} | {int(row['n_scored'])} |"
        )

    if len(errored) > 0:
        lines.append("\n## Submissions with errors\n")
        lines.append("| Andrew ID | Error |")
        lines.append("|-----------|-------|")
        for _, row in errored.iterrows():
            lines.append(f"| {row['andrew_id']} | {row['error']} |")

    return "\n".join(lines) + "\n"


def main():
    if not TRUTH_PATH.exists():
        sys.exit(f"ERROR: truth file not found at {TRUTH_PATH}")
    if not PRED_DIR.exists():
        sys.exit(f"ERROR: predictions directory not found at {PRED_DIR}")

    truth = load_truth()

    pred_files = sorted(PRED_DIR.glob("*.csv"))
    if not pred_files:
        sys.exit(f"No prediction files found in {PRED_DIR}")

    print(f"Scoring {len(pred_files)} submission(s)...")
    results = [score_one(p, truth) for p in pred_files]

    df = pd.DataFrame(results)

    # Sort: scored rows by AUC descending, then errored rows at the bottom
    scored  = df[df["error"] == ""].sort_values("auc", ascending=False)
    errored = df[df["error"] != ""]
    df = pd.concat([scored, errored], ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "leaderboard.csv"
    md_path  = OUT_DIR / "leaderboard.md"

    df.to_csv(csv_path, index=False)
    md_path.write_text(build_markdown_table(df))

    print(f"\nLeaderboard saved to:")
    print(f"  {csv_path}")
    print(f"  {md_path}")
    print()
    print(df[["andrew_id", "auc", "accuracy", "precision", "recall", "f1", "n_scored", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
