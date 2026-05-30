"""
train.py
Train model using feature_extraction.py for both training AND prediction
This guarantees 100% feature consistency
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extraction import extract_features, get_feature_names


def main():
    DATASET_PATH = 'data/phishing_dataset.csv'
    MODEL_PATH = 'models/phishing_model.pkl'
    FEATURES_PATH = 'models/phishing_model_features.pkl'
    SAMPLE_LIMIT = None  # Set to e.g. 50000 for faster testing, None for full

    print("=" * 70)
    print("TRAINING WITH CONSISTENT FEATURE EXTRACTION")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)
    print(f"\nLoaded dataset: {df.shape}")

    urls = df['URL'].tolist()
    raw_labels = df['label'].values

    # PhiUSIIL: 0=Phishing, 1=Legitimate
    # Our convention: 0=Safe, 1=Phishing
    labels = 1 - raw_labels  # flip: 0→1 (Phishing), 1→0 (Safe)

    n_phishing = int((labels == 1).sum())
    n_safe = int((labels == 0).sum())
    print(f"URLs: {len(urls)}")
    print(f"  Phishing (1): {n_phishing}")
    print(f"  Safe (0):     {n_safe}")

    if SAMPLE_LIMIT:
        urls = urls[:SAMPLE_LIMIT]
        labels = labels[:SAMPLE_LIMIT]
        print(f"\n⚠️  Using sample of {SAMPLE_LIMIT} URLs for testing")

    print(f"\nExtracting features from {len(urls)} URLs...")
    feature_names = get_feature_names()
    X_list = []

    for i, url in enumerate(urls):
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(urls)} URLs processed...")
        try:
            feats = extract_features(url)
            X_list.append(feats)
        except Exception as e:
            print(f"  Error at {i}: {url[:50]}... -> {e}")
            X_list.append([0] * len(feature_names))

    X = np.array(X_list)
    y = np.array(labels)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Label vector shape: {y.shape}")

    null_count = np.isnan(X).sum()
    if null_count > 0:
        print(f"Filling {null_count} NaN values with 0")
        X = np.nan_to_num(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Testing set:  {len(X_test)} samples")

    print(f"\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall:    {rec * 100:.2f}%")
    print(f"  F1-Score:  {f1 * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"               Predicted")
    print(f"             Safe  Phishing")
    print(f"Actual Safe  {tn:6d}  {fp:6d}")
    print(f"   Phishing  {fn:6d}  {tp:6d}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(feature_names, FEATURES_PATH)
    print(f"\n Model saved to {MODEL_PATH}")

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"  Model size: {size_mb:.2f} MB")

    print(f"\n{'=' * 70}")
    print("TESTING PREDICTIONS")
    print(f"{'=' * 70}")
    test_urls = [
        "https://www.google.com",
        "https://www.microsoft.com",
        "https://github.com",
        "http://secure-login-paypal.com/verify",
        "http://verify-bank-account.info/password-reset",
        "http://update-crypto-wallet.net/confirm",
        "http://192.168.1.1/admin",
    ]
    for url in test_urls:
        feats = np.array(extract_features(url)).reshape(1, -1)
        pred = rf.predict(feats)[0]
        proba = rf.predict_proba(feats)[0]
        label = "PHISHING" if pred == 1 else "SAFE"
        print(f"  [{label:>8}] (Phish={proba[1]*100:.1f}% Safe={proba[0]*100:.1f}%)  {url}")

    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
