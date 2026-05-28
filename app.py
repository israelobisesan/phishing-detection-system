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

# Initialize Flask app
app = Flask(__name__)

# Load trained model
MODEL_PATH = 'models/phishing_model.pkl'
model = None

def load_model_safely():
    """Load model with better error handling"""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model file not found at {MODEL_PATH}")
            print("Please train the model first by running: python model_training.py")
            return False
        
        model = joblib.load(MODEL_PATH)
        print("✓ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

# Try to load model on startup
load_model_safely()


def predict_url(url):
    """
    Predict if URL is phishing or safe
    """
    if model is None:
        return {
            'error': 'Model not loaded. Please train the model first by running model_training.py',
            'success': False
        }
    
    try:
        # Extract features from URL
        features = extract_features(url)
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        # Prepare result
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
        return {
            'error': str(e),
            'success': False
        }


@app.route('/')
def index():
    """
    Home page
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for URL prediction
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'AI model not available. Please train the model first.'
            }), 503
        
        # Get URL from request
        data = request.get_json()
        url = data.get('url', '').strip()
        
        # Validate URL
        if not url:
            return jsonify({
                'success': False,
                'error': 'Please enter a URL'
            }), 400
        
        # Add http:// if no protocol specified
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # Validate URL format
        if not validators.url(url):
            return jsonify({
                'success': False,
                'error': 'Invalid URL format'
            }), 400
        
        # Make prediction
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
    """
    About page
    """
    return render_template('about.html')


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    API endpoint for batch URL predictions
    """
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
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    }), 200


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500


# Run application
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print("\n" + "="*60)
    print("🛡️  PhishGuard AI - Phishing Detection System")
    print("="*60)
    
    if model is None:
        print("\n⚠️  WARNING: Model not loaded!")
        print("To train the model, run: python model_training.py")
        print("The app will start but predictions won't work until model is trained.\n")
    else:
        print("\n✓ All systems ready!")
        print("✓ Model loaded and ready for predictions\n")
    
    print(f"Starting server on port {port}")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)