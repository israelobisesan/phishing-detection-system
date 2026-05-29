"""
app.py
Main Flask application for phishing detection system
Fixed version with better error handling
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from feature_extraction import extract_features
import validators
import os
import sys

app = Flask(__name__)

MODEL_PATH = 'models/phishing_model.pkl'
MODEL_FEATURES_PATH = 'models/phishing_model_features.pkl'
SCALER_PATH = 'models/scaler.pkl'
model = None
expected_features = None
scaler = None

def get_model_info():
    info = {
        'model_exists': os.path.exists(MODEL_PATH),
        'model_size': os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0,
        'cwd': os.getcwd(),
        'files_in_models': os.listdir('models') if os.path.exists('models') else [],
    }
    return info

def load_model_safely():
    global model, expected_features, scaler
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model file not found at {MODEL_PATH}")
            print(f"CWD: {os.getcwd()}")
            print(f"Files in models/: {os.listdir('models') if os.path.exists('models') else 'models dir missing'}")
            return False

        model = joblib.load(MODEL_PATH)
        print(f"✓ Model loaded successfully! Type: {type(model).__name__}")

        if hasattr(model, 'n_features_in_'):
            print(f"  Expected features: {model.n_features_in_}")
        if hasattr(model, 'classes_'):
            print(f"  Classes: {model.classes_}")

        if os.path.exists(MODEL_FEATURES_PATH):
            expected_features = joblib.load(MODEL_FEATURES_PATH)
            print(f"  Loaded {len(expected_features)} expected feature names")

        if os.path.exists(SCALER_PATH) and os.path.getsize(SCALER_PATH) > 0:
            scaler = joblib.load(SCALER_PATH)
            print("  Scaler loaded")

        return True
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

print("=" * 60)
print("🛡️  PhishGuard AI starting up...")
print("=" * 60)
print(f"  CWD: {os.getcwd()}")
print(f"  Model path: {os.path.abspath(MODEL_PATH)}")
print(f"  Model exists: {os.path.exists(MODEL_PATH)}")
if os.path.exists('models'):
    print(f"  Models dir contents: {os.listdir('models')}")
load_model_safely()
if model:
    print("✓ Model ready for predictions\n")
else:
    print("⚠️  Model not loaded. Predictions will return errors.\n")


def predict_url(url):
    if model is None:
        info = get_model_info()
        return {
            'error': f'Model not loaded. {info}',
            'success': False
        }

    try:
        features = extract_features(url)
        features_array = np.array(features).reshape(1, -1)

        if scaler is not None:
            features_array = scaler.transform(features_array)

        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]

        result = {
            'success': True,
            'url': url,
            'prediction': 'phishing' if prediction == 1 else 'safe',
            'prediction_label': 'Phishing Website' if prediction == 1 else 'Safe Website',
            'confidence': float(max(probability) * 100),
            'phishing_probability': float(probability[1] * 100),
            'safe_probability': float(probability[0] * 100),
            'features_extracted': len(features)
        }

        return result

    except Exception as e:
        import traceback
        return {
            'error': f'Prediction error: {str(e)}',
            'success': False,
            'traceback': traceback.format_exc()
        }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            info = get_model_info()
            return jsonify({
                'success': False,
                'error': f'AI model not available. Model info: {info}'
            }), 503

        data = request.get_json()
        url = data.get('url', '').strip()

        if not url:
            return jsonify({
                'success': False,
                'error': 'Please enter a URL'
            }), 400

        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        if not validators.url(url):
            return jsonify({
                'success': False,
                'error': 'Invalid URL format'
            }), 400

        result = predict_url(url)

        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'AI model not available'
            }), 503

        data = request.get_json()
        urls = data.get('urls', [])

        if not urls or not isinstance(urls, list):
            return jsonify({
                'success': False,
                'error': 'Please provide a list of URLs'
            }), 400

        results = []
        for url in urls:
            url = url.strip()
            if url:
                if not url.startswith(('http://', 'https://')):
                    url = 'http://' + url

                if validators.url(url):
                    result = predict_url(url)
                    results.append(result)

        return jsonify({
            'success': True,
            'total_urls': len(results),
            'results': results
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health():
    info = get_model_info()
    return jsonify({
        'status': 'healthy' if model else 'degraded',
        'model_loaded': model is not None,
        'model_info': info
    }), 200


@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'

    print("\n" + "=" * 60)
    print("🛡️  PhishGuard AI - Phishing Detection System")
    print("=" * 60)

    if model is None:
        print("\n⚠️  WARNING: Model not loaded!")
        print("  The app will start but predictions won't work.")
    else:
        print("\n✓ All systems ready!")
        print("✓ Model loaded and ready for predictions\n")

    print(f"Starting server on port {port}")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=port, debug=debug)
