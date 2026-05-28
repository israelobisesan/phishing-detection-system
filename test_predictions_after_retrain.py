"""
test_predictions_after_retrain.py
Test the retrained model to verify predictions are correct
"""

import joblib
import numpy as np
import pandas as pd
from feature_extraction import extract_features

# Load model
model = joblib.load('models/phishing_model.pkl')

print("=" * 70)
print("TESTING RETRAINED MODEL")
print("=" * 70)

print(f"\n🔍 Model Info:")
print(f"   Classes: {model.classes_}")
print(f"   Class 0 means: ?")
print(f"   Class 1 means: ?")

# Load some training data to verify
df = pd.read_csv('data/phishing_dataset.csv')

print(f"\n📊 Checking Original Dataset Labels:")
print(f"   Original Label 0 count: {(df['label'] == 0).sum()}")
print(f"   Original Label 1 count: {(df['label'] == 1).sum()}")

# Check a known legitimate URL from dataset
legit_sample = df[df['label'] == 1].iloc[0]
print(f"\n✅ Known LEGITIMATE URL from dataset (original label=1):")
print(f"   URL: {legit_sample['URL']}")

# Check a known phishing URL from dataset  
phishing_sample = df[df['label'] == 0].iloc[0]
print(f"\n❌ Known PHISHING URL from dataset (original label=0):")
print(f"   URL: {phishing_sample['URL']}")

print("\n" + "=" * 70)
print("TESTING WITH CLEAR EXAMPLES")
print("=" * 70)

test_cases = [
    ("https://www.google.com", "SAFE"),
    ("https://www.microsoft.com", "SAFE"),
    ("http://secure-paypal-verify.com", "PHISHING"),
    ("http://login-bank-account.info", "PHISHING"),
    ("http://verify-password-update.net", "PHISHING"),
]

for url, expected in test_cases:
    features = extract_features(url)
    features_array = np.array(features).reshape(1, -1)
    
    prediction = model.predict(features_array)[0]
    probabilities = model.predict_proba(features_array)[0]
    
    print(f"\n{'='*70}")
    print(f"URL: {url}")
    print(f"Expected: {expected}")
    print(f"\nModel outputs:")
    print(f"   Raw prediction: {prediction}")
    print(f"   Probabilities: {probabilities}")
    print(f"   probability[0] (class 0): {probabilities[0]*100:.2f}%")
    print(f"   probability[1] (class 1): {probabilities[1]*100:.2f}%")
    
    # Interpretation
    if prediction == 0:
        result = "SAFE"
    else:
        result = "PHISHING"
    
    match = "✓" if result == expected else "✗"
    print(f"\n{match} Interpreted as: {result}")
    print(f"   Safe probability: {probabilities[0]*100:.2f}%")
    print(f"   Phishing probability: {probabilities[1]*100:.2f}%")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

print("\nIf google.com and microsoft.com show as PHISHING:")
print("   → The label interpretation in app.py needs to be flipped")
print("\nIf phishing URLs show as SAFE:")
print("   → The model didn't learn well, need to retrain with better features")