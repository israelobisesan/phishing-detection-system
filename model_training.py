"""
model_training_direct.py
Train Random Forest model using pre-extracted features from PhiUSIIL dataset
This version works with datasets that already have features extracted
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os

def load_and_prepare_data(csv_path):
    """
    Load dataset with pre-extracted features
    """
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    
    df = pd.read_csv(csv_path)
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    
    # Show column names
    print(f"\n📋 Available columns:")
    for i, col in enumerate(df.columns[:10], 1):
        print(f"  {i}. {col}")
    print(f"  ... and {len(df.columns) - 10} more columns")
    
    return df


def prepare_features_and_labels(df):
    """
    Separate features (X) and labels (y) from the dataset
    """
    print("\n" + "=" * 70)
    print("PREPARING FEATURES AND LABELS")
    print("=" * 70)
    
    # The label column should be the last column or named 'label'
    if 'label' in df.columns:
        label_col = 'label'
    elif 'Label' in df.columns:
        label_col = 'Label'
    else:
        # Assume last column is the label
        label_col = df.columns[-1]
        print(f"\n⚠️  Using last column as label: '{label_col}'")
    
    print(f"\n✓ Label column: '{label_col}'")
    
    # Check label distribution
    print(f"\n📊 Label Distribution:")
    label_counts = df[label_col].value_counts()
    for label, count in label_counts.items():
        label_name = "Phishing" if label == 1 else "Legitimate"
        print(f"  {label} ({label_name}): {count:,} samples ({count/len(df)*100:.1f}%)")
    
    # Columns to exclude from features (text columns that can't be used for training)
    exclude_cols = ['URL', 'Domain', 'Title', 'TLD', label_col]
    
    # Select feature columns (all except text columns and label)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Additional check: only include numeric columns
    X_temp = df[feature_cols]
    numeric_cols = X_temp.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < len(feature_cols):
        print(f"\n⚠️  Found {len(feature_cols) - len(numeric_cols)} non-numeric columns, excluding them...")
        non_numeric = set(feature_cols) - set(numeric_cols)
        print(f"  Excluded: {', '.join(non_numeric)}")
        feature_cols = numeric_cols
    
    print(f"\n✓ Using {len(feature_cols)} features")
    print(f"\n📋 Feature columns:")
    for i, col in enumerate(feature_cols[:15], 1):
        print(f"  {i:2d}. {col}")
    if len(feature_cols) > 15:
        print(f"  ... and {len(feature_cols) - 15} more features")
    
    # Create X (features) and y (labels)
    X = df[feature_cols]
    y = df[label_col]
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        print(f"\n⚠️  Found {X.isnull().sum().sum()} missing values - filling with 0")
        X = X.fillna(0)
    
    print(f"\n✓ Feature matrix shape: {X.shape}")
    print(f"✓ Label vector shape: {y.shape}")
    
    return X, y, feature_cols


def train_model(X, y):
    """
    Train Random Forest classifier
    """
    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 70)
    
    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"  Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Testing set:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Initialize Random Forest
    print(f"\n🌲 Random Forest Configuration:")
    print(f"  Number of trees: 100")
    print(f"  Max depth: 20")
    print(f"  Min samples split: 5")
    print(f"  Min samples leaf: 2")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Train model
    print(f"\n🚀 Training started...")
    print("-" * 70)
    rf_model.fit(X_train, y_train)
    print("-" * 70)
    print("✓ Training completed!")
    
    # Make predictions
    print(f"\n🔮 Making predictions on test set...")
    y_pred = rf_model.predict(X_test)
    
    # Evaluate model
    print("\n" + "=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n📈 Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"               Safe  Phishing")
    print(f"Actual Safe    {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"     Phishing  {cm[1][0]:6d}  {cm[1][1]:6d}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (fn + tp)
    
    print(f"\n📊 Additional Metrics:")
    print(f"  True Positives:  {tp:,}")
    print(f"  True Negatives:  {tn:,}")
    print(f"  False Positives: {fp:,}")
    print(f"  False Negatives: {fn:,}")
    print(f"  Specificity:     {specificity:.4f}")
    print(f"  FP Rate:         {false_positive_rate:.4f}")
    print(f"  FN Rate:         {false_negative_rate:.4f}")
    
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    
    return rf_model, accuracy, precision, recall, f1, X_train.columns


def show_feature_importance(model, feature_names):
    """
    Display feature importance
    """
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🏆 Top 15 Most Important Features:")
    print("-" * 70)
    for idx, row in feature_importance.head(15).iterrows():
        print(f"  {row['feature']:30s} : {row['importance']:.6f}")
    
    return feature_importance


def save_model(model, filepath='models/phishing_model.pkl'):
    """
    Save trained model to file
    """
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    print(f"\n💾 Saving model to: {filepath}")
    joblib.dump(model, filepath)
    
    # Verify file was created
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # Convert to MB
        print(f"✓ Model saved successfully!")
        print(f"  File size: {file_size:.2f} MB")
    else:
        print(f"❌ Error: Model file was not created!")


def test_model_with_samples(model, df, feature_cols):
    """
    Test the model with some sample URLs from the dataset
    """
    print("\n" + "=" * 70)
    print("TESTING MODEL WITH SAMPLE URLs")
    print("=" * 70)
    
    # Get some legitimate samples
    legitimate = df[df['label'] == 1].sample(3, random_state=42)
    phishing = df[df['label'] == 0].sample(3, random_state=42)
    
    samples = pd.concat([legitimate, phishing])
    
    for idx, row in samples.iterrows():
        X_sample = row[feature_cols].values.reshape(1, -1)
        prediction = model.predict(X_sample)[0]
        probability = model.predict_proba(X_sample)[0]
        
        actual = "Phishing" if row['label'] == 1 else "Safe"
        predicted = "Phishing" if prediction == 1 else "Safe"
        confidence = max(probability) * 100
        
        match = "✓" if actual == predicted else "✗"
        
        print(f"\n{match} URL: {row['URL'][:60]}...")
        print(f"  Actual: {actual:10s} | Predicted: {predicted:10s} | Confidence: {confidence:.2f}%")


# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🛡️  AI-POWERED PHISHING DETECTION - MODEL TRAINING")
    print("=" * 70)
    
    # Configuration
    DATASET_PATH = 'data/phishing_dataset.csv'
    MODEL_PATH = 'models/phishing_model.pkl'
    
    try:
        # Step 1: Load dataset
        df = load_and_prepare_data(DATASET_PATH)
        
        # Step 2: Prepare features and labels
        X, y, feature_cols = prepare_features_and_labels(df)
        
        # Step 3: Train model
        model, acc, prec, rec, f1, trained_features = train_model(X, y)
        
        # Step 4: Show feature importance
        feature_importance = show_feature_importance(model, trained_features)
        
        # Step 5: Save model
        save_model(model, MODEL_PATH)
        
        # Step 6: Test with samples
        test_model_with_samples(model, df, feature_cols)
        
        # Final summary
        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETE!")
        print("=" * 70)
        print(f"\n📊 Final Results:")
        print(f"  Accuracy:  {acc*100:.2f}%")
        print(f"  Precision: {prec*100:.2f}%")
        print(f"  Recall:    {rec*100:.2f}%")
        print(f"  F1-Score:  {f1*100:.2f}%")
        print(f"\n💾 Model saved to: {MODEL_PATH}")
        print(f"\n🚀 Next step: Run 'python app.py' to start the web application!")
        print("=" * 70 + "\n")
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: Dataset file not found at '{DATASET_PATH}'")
        print("   Please make sure the dataset file exists in the data/ folder")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()