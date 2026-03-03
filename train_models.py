"""
Small training script to create example ML models for the DDoS detection demo.
It will create three models (logistic_DDoS.pkl, rf_DDoS.pkl, gb_DDoS.pkl) trained on
synthetic data matching the features used by the app.

If you have a labeled CSV with the required features and a target column named 'label',
place it next to this script and run: python train_models.py --csv yourfile.csv

This script requires scikit-learn and joblib (already listed in requirements.txt)
"""

import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    'PKT_SIZE', 'NUMBER_OF_PKT', 'NUMBER_OF_BYTE', 'PKT_DELAY_NODE',
    'PKT_RATE', 'BYTE_RATE', 'PKT_AVG_SIZE', 'UTILIZATION', 'PKT_TYPE_ENCODED'
]


def generate_synthetic(n=2000, seed=42):
    rng = np.random.RandomState(seed)
    PKT_SIZE = rng.normal(600, 300, n).clip(40, 1500)
    NUMBER_OF_PKT = rng.poisson(20, n).clip(1, None)
    NUMBER_OF_BYTE = (PKT_SIZE * NUMBER_OF_PKT).astype(float)
    PKT_DELAY_NODE = rng.exponential(0.05, n)
    PKT_RATE = rng.normal(5, 10, n).clip(0, None)
    BYTE_RATE = PKT_RATE * PKT_SIZE
    PKT_AVG_SIZE = PKT_SIZE
    UTILIZATION = (NUMBER_OF_PKT / 100.0)
    PKT_TYPE_ENCODED = rng.choice([0,1,2,3], size=n, p=[0.2,0.2,0.1,0.5])

    # Simple rule: high pkt_rate -> attack label. Also inject a small fraction of labelled attacks
    label = ((PKT_RATE > 100) | (rng.rand(n) < 0.06)).astype(int)

    df = pd.DataFrame({
        'PKT_SIZE': PKT_SIZE,
        'NUMBER_OF_PKT': NUMBER_OF_PKT,
        'NUMBER_OF_BYTE': NUMBER_OF_BYTE,
        'PKT_DELAY_NODE': PKT_DELAY_NODE,
        'PKT_RATE': PKT_RATE,
        'BYTE_RATE': BYTE_RATE,
        'PKT_AVG_SIZE': PKT_AVG_SIZE,
        'UTILIZATION': UTILIZATION,
        'PKT_TYPE_ENCODED': PKT_TYPE_ENCODED,
        'label': label
    })
    return df


def train_and_save(X, y, model, name):
    print(f"Training {name}...")
    # 'model' is expected to be a Pipeline (scaler + estimator) passed from caller.
    # Fit the pipeline that will be saved so all steps are trained.
    model.fit(X, y)
    # Save the fitted pipeline along with expected feature names
    payload = {'pipeline': model, 'feature_names': FEATURE_NAMES}
    joblib.dump(payload, f"{name}.pkl")
    print(f"Saved {name}.pkl (fitted pipeline + feature names)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='Path to CSV with features and label column', default=None)
    parser.add_argument('--samples', type=int, default=3000)
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        if 'label' not in df.columns:
            raise SystemExit('CSV must contain a "label" column (0/1)')
    else:
        df = generate_synthetic(n=args.samples)

    X = df[FEATURE_NAMES]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        (LogisticRegression(max_iter=400, solver='liblinear'), 'logistic_DDoS'),
        (RandomForestClassifier(n_estimators=120, max_depth=None, random_state=42), 'rf_DDoS'),
        (GradientBoostingClassifier(n_estimators=80, learning_rate=0.08, random_state=42), 'gb_DDoS'),
        (ExtraTreesClassifier(n_estimators=160, random_state=42), 'et_DDoS'),
        (KNeighborsClassifier(n_neighbors=5, weights='distance'), 'knn_DDoS'),
        (MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=400, random_state=42, early_stopping=True), 'mlp_DDoS')
    ]

    for m, name in models:
        # Build a pipeline: scaler + estimator to ensure consistent preprocessing at predict time
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', m)])
        train_and_save(X_train, y_train, pipe, name)
        preds = pipe.predict(X_test)
        print(f"--- {name} evaluation ---")
        print(classification_report(y_test, preds))

    print("Training complete. Generated model files in the current folder.")
