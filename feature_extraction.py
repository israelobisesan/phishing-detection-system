"""
feature_extraction.py
Extract ONLY URL-based features (26 features) to match the retrained model
No HTML features - only what we can extract from the URL itself
"""

import re
import tldextract
from urllib.parse import urlparse

def extract_features(url):
    """
    Extract 26 URL-only features matching the retrained model
    Returns features in the exact order the model expects
    """
    features = []
    
    try:
        # Parse URL
        parsed = urlparse(url)
        ext = tldextract.extract(url)
        
        # Feature 1: URLLength
        url_length = len(url)
        features.append(url_length)
        
        # Feature 2: DomainLength
        domain = ext.domain + '.' + ext.suffix if ext.suffix else ext.domain
        domain_length = len(domain)
        features.append(domain_length)
        
        # Feature 3: IsDomainIP (1 if IP address, 0 otherwise)
        ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        netloc = parsed.netloc.split(':')[0]  # Remove port if present
        is_domain_ip = 1 if re.match(ip_pattern, netloc) else 0
        features.append(is_domain_ip)
        
        # Feature 4: URLSimilarityIndex (percentage of alphanumeric chars)
        alphanumeric_count = sum(c.isalnum() for c in url)
        url_similarity_index = (alphanumeric_count / url_length * 100) if url_length > 0 else 0
        features.append(url_similarity_index)
        
        # Feature 5: CharContinuationRate (ratio of repeated consecutive chars)
        continuation_count = sum(1 for i in range(len(url)-1) if url[i] == url[i+1])
        char_continuation_rate = continuation_count / (url_length - 1) if url_length > 1 else 0
        features.append(char_continuation_rate)
        
        # Feature 6: TLDLegitimateProb (probability based on common TLDs)
        common_tlds = {
            'com': 0.5229071, 'org': 0.0799628, 'net': 0.04, 'edu': 0.03,
            'gov': 0.02, 'co': 0.02, 'uk': 0.028555, 'de': 0.0326503,
            'jp': 0.02, 'fr': 0.02, 'in': 0.0050842, 'ca': 0.02,
            'au': 0.02, 'ru': 0.0180132, 'nl': 0.02, 'br': 0.02,
            'it': 0.02, 'es': 0.02, 'se': 0.02, 'ch': 0.02,
            'us': 0.02, 'pl': 0.02, 'ie': 0.0015878, 'hu': 0.0022172
        }
        tld = ext.suffix.lower() if ext.suffix else ''
        tld_prob = common_tlds.get(tld, 0.01)
        features.append(tld_prob)
        
        # Feature 7: URLCharProb (entropy-like measure)
        unique_chars = len(set(url))
        url_char_prob = unique_chars / url_length if url_length > 0 else 0
        features.append(url_char_prob)
        
        # Feature 8: TLDLength
        tld_length = len(ext.suffix) if ext.suffix else 0
        features.append(tld_length)
        
        # Feature 9: NoOfSubDomain
        subdomain = ext.subdomain
        num_subdomains = subdomain.count('.') + 1 if subdomain else 0
        features.append(num_subdomains)
        
        # Feature 10: HasObfuscation
        has_obfuscation = 1 if '%' in url or any(x in url for x in ['\\x', '\\u']) else 0
        features.append(has_obfuscation)
        
        # Feature 11: NoOfObfuscatedChar
        obfuscated_chars = url.count('%')
        features.append(obfuscated_chars)
        
        # Feature 12: ObfuscationRatio
        obfuscation_ratio = obfuscated_chars / url_length if url_length > 0 else 0
        features.append(obfuscation_ratio)
        
        # Feature 13: NoOfLettersInURL
        num_letters = sum(c.isalpha() for c in url)
        features.append(num_letters)
        
        # Feature 14: LetterRatioInURL
        letter_ratio = num_letters / url_length if url_length > 0 else 0
        features.append(letter_ratio)
        
        # Feature 15: NoOfDegitsInURL (digits)
        num_digits = sum(c.isdigit() for c in url)
        features.append(num_digits)
        
        # Feature 16: DegitRatioInURL (digit ratio)
        digit_ratio = num_digits / url_length if url_length > 0 else 0
        features.append(digit_ratio)
        
        # Feature 17: NoOfEqualsInURL
        num_equals = url.count('=')
        features.append(num_equals)
        
        # Feature 18: NoOfQMarkInURL
        num_qmark = url.count('?')
        features.append(num_qmark)
        
        # Feature 19: NoOfAmpersandInURL
        num_ampersand = url.count('&')
        features.append(num_ampersand)
        
        # Feature 20: NoOfOtherSpecialCharsInURL
        special_chars = sum(1 for c in url if not c.isalnum() and c not in ['.', '/', ':', '-', '_', '?', '=', '&', '%'])
        features.append(special_chars)
        
        # Feature 21: SpacialCharRatioInURL (special character ratio)
        special_char_ratio = special_chars / url_length if url_length > 0 else 0
        features.append(special_char_ratio)
        
        # Feature 22: IsHTTPS
        is_https = 1 if parsed.scheme == 'https' else 0
        features.append(is_https)
        
        # Feature 23: HasPasswordField (check URL for keywords)
        has_password = 1 if any(word in url.lower() for word in ['password', 'passwd', 'pwd', 'pass']) else 0
        features.append(has_password)
        
        # Feature 24: Bank (bank-related keywords)
        has_bank = 1 if any(word in url.lower() for word in ['bank', 'banking', 'banker']) else 0
        features.append(has_bank)
        
        # Feature 25: Pay (payment-related keywords)
        has_pay = 1 if any(word in url.lower() for word in ['pay', 'payment', 'paypal', 'checkout']) else 0
        features.append(has_pay)
        
        # Feature 26: Crypto (crypto-related keywords)
        has_crypto = 1 if any(word in url.lower() for word in ['crypto', 'bitcoin', 'wallet', 'coin']) else 0
        features.append(has_crypto)
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {url}: {e}")
        # Return default features (26 zeros)
        return [0] * 26


def get_feature_names():
    """
    Returns the exact 26 feature names in order
    """
    return [
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
        'HasPasswordField',
        'Bank',
        'Pay',
        'Crypto'
    ]


# Test the function
if __name__ == "__main__":
    print("=" * 70)
    print("TESTING FEATURE EXTRACTION (26 URL-ONLY FEATURES)")
    print("=" * 70)
    
    # Test with sample URLs
    test_urls = [
        "https://www.google.com",
        "http://secure-login-paypal.com/verify-account",
        "https://www.microsoft.com",
        "http://192.168.1.1/admin",
        "http://verify-bank-account.info/password-reset",
        "http://update-crypto-wallet.net/confirm"
    ]
    
    for url in test_urls:
        features = extract_features(url)
        print(f"\n{'='*70}")
        print(f"URL: {url}")
        print(f"Total features: {len(features)}")
        print(f"\nKey indicators:")
        print(f"  URL Length: {features[0]}")
        print(f"  Domain Length: {features[1]}")
        print(f"  HTTPS: {'Yes' if features[21] else 'No'}")
        print(f"  Has 'password': {'Yes' if features[22] else 'No'}")
        print(f"  Has 'bank': {'Yes' if features[23] else 'No'}")
        print(f"  Has 'pay': {'Yes' if features[24] else 'No'}")
        print(f"  Has 'crypto': {'Yes' if features[25] else 'No'}")
    
    print("\n" + "=" * 70)
    print(f"✓ Feature extraction returns exactly {len(extract_features('https://test.com'))} features")
    print("✓ This matches the retrained model's expectations!")
    print("=" * 70)