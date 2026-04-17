"""
Standalone script to run the lasso tuning job from a terminal session.
This is the script you run inside a screen session (Section 6).

Pass your C grid as space-separated values on the command line:

    screen -S lasso_fit
    source .venv/bin/activate
    python src/run_slow_fit.py 0.0001 0.001 0.01 0.1 1.0

Results are saved to results_lasso.json in the project root.
"""

import argparse
import json
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Single-threaded to avoid placing strain on the shared server
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

TRAIN_PATH = Path('/home/mlfp_shared/nij_data/train_preprocessed.csv')
OUTCOME    = 'Recidivism_Within_3years'
OUT_PATH   = Path('results_lasso.json')


def main():
    parser = argparse.ArgumentParser(
        description='Tune a lasso logistic regression over a grid of alpha values.'
    )
    parser.add_argument(
        'C_values', nargs='+', type=float,
        metavar='C',
        help='Regularization strengths to try (1–10 values, e.g. 0.0001 0.001 0.01 0.1 1.0)',
    )
    args = parser.parse_args()
    C_grid = args.C_values

    assert len(C_grid) <= 10, (
        f'You provided {len(C_grid)} C values — keep it to <= 10 on the shared server.'
    )

    start = time.time()

    print('Loading data...')
    train = pd.read_csv(TRAIN_PATH)
    feature_cols = [c for c in train.columns if c != OUTCOME]
    X = train[feature_cols].values
    y = train[OUTCOME].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=10
    )

    print(f'Fitting lasso grid: {C_grid}')
    results = []
    best_auc = -1
    best_C   = None

    for C in C_grid:
        model = LogisticRegression(
            solver='saga', l1_ratio=1, C=C,
            max_iter=5000, random_state=10
        )
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_val)[:, 1]
        pred = model.predict(X_val)
        auc  = roc_auc_score(y_val, prob)
        acc  = accuracy_score(y_val, pred)
        f1   = f1_score(y_val, pred)
        results.append({
            'C':            C,
            'val_auc':      round(auc, 4),
            'val_accuracy': round(acc, 4),
            'val_f1':       round(f1,  4),
        })
        print(f'  C={C:.4f}  AUC={auc:.4f}  Acc={acc:.4f}  F1={f1:.4f}')
        if auc > best_auc:
            best_auc = auc
            best_C   = C

    print("\nModel fitting complete")
    stop = time.time()
    print(f'\nBest C: {best_C}  (AUC={best_auc:.4f})')
    print('\nSaving results...')

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f'Results saved to {OUT_PATH}')


if __name__ == '__main__':
    main()
