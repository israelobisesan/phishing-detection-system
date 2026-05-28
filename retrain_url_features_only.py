"""
retrain_url_features_only.py
Retrain the model using ONLY URL-based features (no HTML features)
This ensures predictions work without needing to fetch webpage content
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os

def load_and_prepare_data(csv_path):
    """Load dataset"""
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    
    df = pd.read_csv(csv_path)
    print(f"\n✓ Dataset loaded: {df.shape}")
    return df

def select_url_only_features(df):
    """
    Select ONLY features that can be extracted from URLs
    (no HTML/content-based features)
    """
    print("\n" + "=" * 70)
    print("SELECTING URL-ONLY FEATURES")
    print("=" * 70)
    
    # Features we CAN extract from URL alone
    url_features = [
        'URLLength',
        'DomainLength',
        'IsDomainIP',
        'URLSimilarityIndex',
        'CharContinuationRate',
        'TLDLegitimateProb',
        'URLCharProb',
        'TLDLength',
        'NoOfSubDomain',
        'HasObfuscation',
        'NoOfObfuscatedChar',
        'ObfuscationRatio',
        'NoOfLettersInURL',
        'LetterRatioInURL',
        'NoOfDegitsInURL',
        'DegitRatioInURL',
        'NoOfEqualsInURL',
        'NoOfQMarkInURL',
        'NoOfAmpersandInURL',
        'NoOfOtherSpecialCharsInURL',
        'SpacialCharRatioInURL',
        'IsHTTPS',
        # These can be detected from URL keywords
        'HasPasswordField',
        'Bank',
        'Pay',
        'Crypto'
    ]
    
    # Check which features exist in the dataset
    available_features = [f for f in url_features if f in df.columns]
    
    print(f"\n✓ Selected {len(available_features)} URL-only features:")
    for i, feature in enumerate(available_features, 1):
        print(f"   {i:2d}. {feature}")
    
    # Get features and labels
    X = df[available_features]
    y = df['label']
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        print(f"\n⚠️  Filling {X.isnull().sum().sum()} missing values with 0")
        X = X.fillna(0)
    
    # IMPORTANT: In PhiUSIIL dataset, labels are: 0=Phishing, 1=Legitimate
    # We need to flip them to match our convention: 0=Safe, 1=Phishing
    print(f"\n⚠️  FLIPPING LABELS (PhiUSIIL has 0=Phishing, 1=Safe)")
    y = 1 - y  # Flip: 0→1, 1→0
    
    print(f"\n📊 Label Distribution (after flipping):")
    print(f"   Phishing (1): {(y == 1).sum():,} ({(y == 1).sum()/len(y)*100:.1f}%)")
    print(f"   Safe (0):     {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.1f}%)")
    
    return X, y, available_features

def train_model(X, y):
    """Train Random Forest on URL-only features"""
    print("\n" + "=" * 70)
    print("TRAINING MODEL (URL FEATURES ONLY)")
    print("=" * 70)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {len(X_train):,} samples")
    print(f"   Testing:  {len(X_test):,} samples")
    
    # Train with balanced class weights to handle imbalance
    print(f"\n🌲 Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=150,           # More trees for better performance
        max_depth=25,               # Slightly deeper trees
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',    # Handle class imbalance
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)
    
    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1-Score:  {f1*100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📊 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"               Safe  Phishing")
    print(f"Actual Safe    {tn:6d}  {fp:6d}")
    print(f"     Phishing  {fn:6d}  {tp:6d}")
    
    print(f"\n📊 Additional Metrics:")
    print(f"   True Positives:  {tp:,}")
    print(f"   True Negatives:  {tn:,}")
    print(f"   False Positives: {fp:,}")
    print(f"   False Negatives: {fn:,}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))
    
    return rf_model, accuracy, precision, recall, f1, X_train.columns

def show_feature_importance(model, feature_names):
    """Show which features matter most"""
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n🏆 Top Features:")
    for idx, row in importance_df.iterrows():
        bar_length = int(row['Importance'] * 50)
        bar = '█' * bar_length
        print(f"   {row['Feature']:30s} {bar} {row['Importance']:.4f}")
    
    return importance_df

def test_predictions(model, feature_cols):
    """Test with actual phishing and safe URLs"""
    print("\n" + "=" * 70)
    print("TESTING WITH SAMPLE URLs")
    print("=" * 70)
    
    from feature_extraction import extract_features
    
    test_cases = [
        ("https://www.google.com", 0, "Safe"),
        ("https://www.microsoft.com", 0, "Safe"),
        ("http://secure-paypal-login.com", 1, "Phishing"),
        ("http://verify-account-bank.info", 1, "Phishing"),
        ("http://update-password.net/confirm", 1, "Phishing"),
    ]
    
    for url, expected, expected_label in test_cases:
        # Extract features
        features = extract_features(url)
        
        # Get only the features used in training
        feature_dict = dict(zip([
            'URLLength', 'DomainLength', 'IsDomainIP', 'URLSimilarityIndex',
            'CharContinuationRate', 'TLDLegitimateProb', 'URLCharProb', 'TLDLength',
            'NoOfSubDomain', 'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio',
            'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
            'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
            'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS',
            'HasPasswordField', 'Bank', 'Pay', 'Crypto'
        ], features[:26]))
        
        X_test = pd.DataFrame([feature_dict])[feature_cols]
        
        prediction = model.predict(X_test)[0]
        probabilities = model.predict_proba(X_test)[0]
        
        predicted_label = "Phishing" if prediction == 1 else "Safe"
        match = "✓" if prediction == expected else "✗"
        
        print(f"\n{match} URL: {url}")
        print(f"   Expected: {expected_label:10s} | Predicted: {predicted_label:10s}")
        print(f"   Probabilities: Safe={probabilities[0]*100:.1f}% | Phishing={probabilities[1]*100:.1f}%")

def save_model(model, feature_cols, filepath='models/phishing_model.pkl'):
    """Save model and feature list"""
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save model
    joblib.dump(model, filepath)
    
    # Save feature columns list
    feature_file = filepath.replace('.pkl', '_features.pkl')
    joblib.dump(list(feature_cols), feature_file)
    
    print(f"\n✓ Model saved to: {filepath}")
    print(f"✓ Features saved to: {feature_file}")
    
    file_size = os.path.getsize(filepath) / (1024 * 1024)
    print(f"✓ Model size: {file_size:.2f} MB")

# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🛡️  RETRAIN WITH URL-ONLY FEATURES")
    print("=" * 70)
    
    try:
        # Load data
        df = load_and_prepare_data('data/phishing_dataset.csv')
        
        # Select URL-only features
        X, y, feature_cols = select_url_only_features(df)
        
        # Train model
        model, acc, prec, rec, f1, trained_features = train_model(X, y)
        
        # Feature importance
        importance_df = show_feature_importance(model, trained_features)
        
        # Test predictions
        test_predictions(model, trained_features)
        
        # Save model
        save_model(model, trained_features)
        
        print("\n" + "=" * 70)
        print("✓ RETRAINING COMPLETE!")
        print("=" * 70)
        print(f"\n📊 Final Results:")
        print(f"   Accuracy:  {acc*100:.2f}%")
        print(f"   Precision: {prec*100:.2f}%")
        print(f"   Recall:    {rec*100:.2f}%")
        print(f"   F1-Score:  {f1*100:.2f}%")
        print(f"\n🚀 Now run: python app.py")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()