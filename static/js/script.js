const urlForm = document.getElementById('urlForm');
const urlInput = document.getElementById('urlInput');
const checkBtn = document.getElementById('checkBtn');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultCard = document.getElementById('resultCard');
const errorCard = document.getElementById('errorCard');
const resultIcon = document.getElementById('resultIcon');
const resultTitle = document.getElementById('resultTitle');
const checkedUrl = document.getElementById('checkedUrl');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceProgress = document.getElementById('confidenceProgress');
const safeProbability = document.getElementById('safeProbability');
const phishingProbability = document.getElementById('phishingProbability');
const recommendation = document.getElementById('recommendation');
const errorMessage = document.getElementById('errorMessage');
const examplesList = document.getElementById('examplesList');

const exampleUrls = [
  'https://www.google.com',
  'http://secure-login-paypal.com/verify',
  'https://www.microsoft.com'
];

exampleUrls.forEach(url => {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'example-btn';
  btn.textContent = url.length > 32 ? url.slice(0, 30) + '...' : url;
  btn.addEventListener('click', () => {
    urlInput.value = url;
    urlInput.focus();
    urlForm.dispatchEvent(new Event('submit'));
  });
  examplesList.appendChild(btn);
});

urlForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  const url = urlInput.value.trim();

  if (!url) {
    urlInput.focus();
    urlInput.style.borderColor = 'var(--danger)';
    setTimeout(() => {
      urlInput.style.borderColor = '';
    }, 1500);
    return;
  }

  hideAllCards();
  showLoading();

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });

    const data = await response.json();
    hideLoading();

    if (data.success) {
      showResult(data);
    } else {
      showError(data.error || 'Could not analyze this URL. Please try again.');
    }
  } catch (error) {
    hideLoading();
    showError('Failed to connect to the server. Please check your connection.');
    console.error('Error:', error);
  }
});

function showLoading() {
  loadingSpinner.style.display = 'flex';
  checkBtn.disabled = true;
  checkBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
}

function hideLoading() {
  loadingSpinner.style.display = 'none';
  checkBtn.disabled = false;
  checkBtn.innerHTML = '<i class="fas fa-shield-halved"></i> Check URL';
}

function hideAllCards() {
  resultCard.style.display = 'none';
  errorCard.style.display = 'none';
}

function showResult(data) {
  requestAnimationFrame(() => {
    checkedUrl.textContent = data.url;

    const confidence = Math.round(data.confidence * 10) / 10;
    confidenceValue.textContent = confidence + '%';
    confidenceProgress.style.width = confidence + '%';

    safeProbability.textContent = (data.safe_probability).toFixed(1) + '%';
    phishingProbability.textContent = (data.phishing_probability).toFixed(1) + '%';

    const isPhishing = data.prediction === 'phishing';

    resultCard.className = 'glass-card result-card ' + (isPhishing ? 'phishing' : 'safe');
    resultIcon.className = 'result-icon ' + (isPhishing ? 'phishing' : 'safe');
    resultTitle.textContent = isPhishing ? 'Phishing Detected!' : 'URL is Safe';

    recommendation.className = 'recommendation ' + (isPhishing ? 'phishing' : 'safe');

    if (isPhishing) {
      recommendation.innerHTML = `
        <strong><i class="fas fa-triangle-exclamation"></i> Warning</strong>
        <p>This URL exhibits patterns consistent with phishing attacks. Do not enter any personal information, passwords, or payment details. Close the website immediately if you have it open.</p>
      `;
    } else {
      recommendation.innerHTML = `
        <strong><i class="fas fa-circle-check"></i> All Clear</strong>
        <p>This URL appears to be legitimate and safe to visit. However, always exercise caution when entering sensitive information online.</p>
      `;
    }

    resultCard.style.display = 'block';

    setTimeout(() => {
      resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);
  });
}

function showError(message) {
  errorMessage.textContent = message;
  errorCard.style.display = 'block';

  setTimeout(() => {
    errorCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, 100);
}

function resetForm() {
  urlInput.value = '';
  urlInput.style.borderColor = '';
  hideAllCards();
  hideLoading();
  urlInput.focus();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

urlInput.addEventListener('input', () => {
  const value = urlInput.value.trim();
  if (value.length === 0) {
    urlInput.className = '';
    return;
  }
  const hasDot = value.includes('.');
  const hasProtocol = /^https?:\/\//i.test(value);
  const isValid = /^(https?:\/\/)?([\w\-]+\.)+[\w\-]{2,}/i.test(value);
  if (isValid) {
    urlInput.className = 'valid';
  } else if (hasProtocol || hasDot) {
    urlInput.className = 'invalid';
  } else {
    urlInput.className = '';
  }
});

window.addEventListener('load', () => {
  urlInput.focus();
});

urlInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    urlForm.dispatchEvent(new Event('submit'));
  }
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    resetForm();
  }
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    e.preventDefault();
    urlInput.focus();
    urlInput.select();
  }
});

console.log('%c PhishGuard AI ', 'background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; padding: 8px 16px; border-radius: 4px; font-size: 16px; font-weight: bold;');
console.log('%c Protecting users from phishing threats ', 'color: #22c55e; font-size: 13px;');
