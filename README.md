# PhishGuard AI

AI-powered phishing detection system. Enter a URL and get instant analysis.

## Run locally

```bash
pip install -r requirements.txt
python model_training.py
python app.py
```

## Deploy on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Build: `pip install -r requirements.txt`
5. Start: `gunicorn app:app`
