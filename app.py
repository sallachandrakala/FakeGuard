"""
=============================================================================
  FAKE NEWS DETECTION - FLASK WEB APPLICATION
  File: app.py
  Author: [Your Name]
  Description:
      Main Flask app that serves the web interface, loads the trained ML model,
      preprocesses user input, and returns predictions via REST API.
=============================================================================
"""

import os
import re
import pickle
import datetime
import json
import hashlib

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from flask import (Flask, render_template, request,
                   jsonify, redirect, url_for, session, flash)

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# =============================================================================
#  FLASK APP INITIALIZATION
# =============================================================================

app = Flask(__name__)
app.secret_key = 'fakenews_secret_key_2024'   # Required for session storage

# =============================================================================
#  USER STORAGE (Simple JSON file for demo purposes)
# =============================================================================

USERS_FILE = 'users.json'
HISTORY_FILE = 'prediction_history.json'

def load_users():
    """Load users from JSON file."""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    """Save users to JSON file."""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def load_history():
    """Load prediction history from JSON file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history):
    """Save prediction history to JSON file."""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def hash_password(password):
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

# =============================================================================
#  LOAD TRAINED MODEL (runs once at startup)
# =============================================================================

MODEL_PATH      = 'model/best_model.pkl'
VECTORIZER_PATH = 'model/vectorizer.pkl'
METADATA_PATH   = 'model/metadata.pkl'

model      = None
vectorizer = None
metadata   = {}

def load_model():
    """Loads the saved ML model, vectorizer, and metadata from disk."""
    global model, vectorizer, metadata
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(VECTORIZER_PATH, 'rb') as f:
                vectorizer = pickle.load(f)
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, 'rb') as f:
                    metadata = pickle.load(f)
            print("[INFO] ✅ Model loaded successfully.")
            return True
        else:
            print("[WARNING] Model files not found. Run model_training.py first.")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False

# Load model on startup
model_loaded = load_model()


# =============================================================================
#  TEXT PREPROCESSING FUNCTION
# =============================================================================

def preprocess_text(text):
    """
    Same preprocessing pipeline used during training.
    Ensures consistency between training and prediction.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    return ' '.join(tokens)


# =============================================================================
#  PREDICTION LOGIC
# =============================================================================

def is_gibberish_or_meaningless(text, clean_text):
    """
    Detects if the input text is gibberish, random typing, or meaningless.
    Returns True if the text appears to be fake/gibberish.
    """
    # Check if cleaned text is too short (indicates mostly special chars/numbers)
    if len(clean_text.strip()) < 3:
        return True
    
    # Check ratio of special characters to letters in original text
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars > 0 and special_chars / alpha_chars > 0.3:
        return True
    
    # Check if text has very low vocabulary diversity (repeated characters)
    words = clean_text.split()
    if len(words) > 0:
        unique_words = len(set(words))
        if unique_words < len(words) * 0.3 and len(words) > 3:
            return True
    
    # Check for consecutive repeated characters (like "aaaaa" or "jjjj")
    if any(c * 3 in text.lower() for c in 'abcdefghijklmnopqrstuvwxyz'):
        return True
    
    # Check for keyboard patterns (consecutive keys on QWERTY keyboard)
    keyboard_rows = [
        'qwertyuiop',
        'asdfghjkl',
        'zxcvbnm'
    ]
    text_lower = text.lower().replace('[', '').replace(']', '')
    for row in keyboard_rows:
        for i in range(len(row) - 3):
            pattern = row[i:i+4]
            if pattern in text_lower:
                return True
    
    # Check for very low vowel ratio (indicates random consonants)
    vowels = 'aeiou'
    alpha_only = ''.join(c for c in text_lower if c.isalpha())
    if len(alpha_only) > 5:
        vowel_count = sum(1 for c in alpha_only if c in vowels)
        if vowel_count / len(alpha_only) < 0.15:  # Less than 15% vowels
            return True
    
    # Check for character randomness (high entropy-like pattern)
    # If characters alternate too much between different keyboard regions
    if len(alpha_only) > 10:
        left_hand = 'qwertasdfgzxcv'
        right_hand = 'yuiophjklbnm'
        alternations = 0
        for i in range(len(alpha_only) - 1):
            curr_left = alpha_only[i] in left_hand
            next_left = alpha_only[i+1] in left_hand
            if curr_left != next_left:
                alternations += 1
        if alternations / len(alpha_only) > 0.7:  # Too much alternating
            return True
    
    return False


def predict_news(text):
    """
    Takes raw news text, preprocesses it, vectorizes it,
    and returns prediction with confidence scores.

    Returns:
        dict with keys: label, confidence, fake_prob, real_prob, word_count, explanation
    """
    if not model_loaded or model is None or vectorizer is None:
        return None

    # Preprocess
    clean_text = preprocess_text(text)

    if not clean_text.strip():
        return None
    
    # Check for gibberish/meaningless text before ML prediction
    if is_gibberish_or_meaningless(text, clean_text):
        return {
            'label': 'Fake',
            'confidence': 95.0,
            'fake_prob': 95.0,
            'real_prob': 5.0,
            'word_count': len(text.split()),
            'model_used': 'Gibberish Filter + ' + metadata.get('model_name', 'ML Model'),
            'explanation': {
                'reason': 'This text appears to be gibberish, random typing, or meaningless content with no coherent news structure.',
                'key_words': []
            }
        }

    # Vectorize
    text_tfidf = vectorizer.transform([clean_text])

    # Predict
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]

    fake_prob = round(float(probabilities[0]) * 100, 2)
    real_prob = round(float(probabilities[1]) * 100, 2)
    confidence = max(fake_prob, real_prob)
    label = "Real" if prediction == 1 else "Fake"

    # Extract feature importance for explanation
    explanation = get_feature_explanation(text_tfidf, clean_text, prediction)

    return {
        'label':      label,
        'confidence': round(confidence, 2),
        'fake_prob':  fake_prob,
        'real_prob':  real_prob,
        'word_count': len(text.split()),
        'model_used': metadata.get('model_name', 'ML Model'),
        'explanation': explanation
    }


def get_feature_explanation(text_tfidf, clean_text, prediction):
    """
    Extracts important features (words) that contributed to the prediction.
    Returns a list of top influential words with their importance scores.
    """
    try:
        # Get feature names from vectorizer
        feature_names = vectorizer.get_feature_names_out()
        
        # Get coefficients from the model (for Logistic Regression)
        if hasattr(model, 'coef_'):
            coef = model.coef_[0]
        else:
            # For other models, use feature importance if available
            if hasattr(model, 'feature_importances_'):
                coef = model.feature_importances_
            else:
                return {"reason": "Model does not support feature importance extraction", "key_words": []}
        
        # Get the TF-IDF values for this specific text
        tfidf_values = text_tfidf.toarray()[0]
        
        # Calculate importance: coefficient * tfidf value
        importance_scores = coef * tfidf_values
        
        # Get indices of non-zero features (words actually present in text)
        non_zero_indices = [i for i in range(len(tfidf_values)) if tfidf_values[i] > 0]
        
        # Sort by absolute importance
        sorted_indices = sorted(non_zero_indices, key=lambda i: abs(importance_scores[i]), reverse=True)
        
        # Get top 5-10 influential words
        top_words = []
        for idx in sorted_indices[:8]:
            word = feature_names[idx]
            score = importance_scores[idx]
            top_words.append({
                'word': word,
                'importance': round(float(score), 4),
                'direction': 'fake' if score < 0 else 'real'
            })
        
        # Generate explanation text
        if prediction == 1:  # Real
            reason = "This article is classified as REAL because it contains language patterns and vocabulary commonly found in credible news sources."
        else:  # Fake
            reason = "This article is classified as FAKE because it contains language patterns and vocabulary commonly associated with sensational or misleading content."
        
        return {
            "reason": reason,
            "key_words": top_words
        }
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return {"reason": "Could not extract feature importance", "key_words": []}


# =============================================================================
#  SHARED PREDICTION HISTORY (JSON FILE)
# =============================================================================

def get_history():
    """Returns prediction history from shared JSON file."""
    return load_history()

def add_to_history(text, result):
    """Adds a new prediction entry to shared history file."""
    history = load_history()
    
    # Get current user info if logged in
    username = session.get('username', 'Anonymous')
    user_name = session.get('name', username)  # Use full name if available, else username
    user_email = session.get('email', 'N/A')

    entry = {
        'id':         len(history) + 1,
        'text':       text[:200] + '...' if len(text) > 200 else text,
        'full_text':  text,
        'label':      result['label'],
        'confidence': result['confidence'],
        'fake_prob':  result['fake_prob'],
        'real_prob':  result['real_prob'],
        'word_count': result['word_count'],
        'timestamp':  datetime.datetime.now().strftime('%d %b %Y, %I:%M %p'),
        'model_used': result.get('model_used', 'ML Model'),
        'username':   user_name,  # Display full name
        'email':      user_email
    }
    history.insert(0, entry)        # Newest first
    history = history[:100]         # Keep last 100 entries
    save_history(history)
    return entry


# =============================================================================
#  FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    """
    HOME / DASHBOARD PAGE
    Renders the main detection interface.
    Shows model status and quick stats.
    """
    all_history = get_history()
    stats = {
        'total':     len(all_history),
        'fake':      sum(1 for h in all_history if h['label'] == 'Fake'),
        'real':      sum(1 for h in all_history if h['label'] == 'Real'),
        'model_ok':  model_loaded,
        'model_name': metadata.get('model_name', 'Not Loaded')
    }
    return render_template('index.html', stats=stats)


@app.route('/predict', methods=['POST'])
def predict():
    """
    PREDICTION API ENDPOINT
    Accepts POST with JSON or form data containing 'news_text'.
    Returns JSON with prediction result.
    """
    # Accept both JSON and form data
    if request.is_json:
        data = request.get_json()
        news_text = data.get('news_text', '').strip()
    else:
        news_text = request.form.get('news_text', '').strip()

    # Validation
    if not news_text:
        return jsonify({'error': 'Please enter some news text.'}), 400

    if len(news_text) < 10:
        return jsonify({'error': 'Text too short. Please enter at least 10 characters.'}), 400

    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please run model_training.py first.'
        }), 500

    # Predict
    result = predict_news(news_text)

    if result is None:
        return jsonify({'error': 'Could not process the text. Please try again.'}), 500

    # Save to history
    entry = add_to_history(news_text, result)

    return jsonify({
        'success':    True,
        'label':      result['label'],
        'confidence': result['confidence'],
        'fake_prob':  result['fake_prob'],
        'real_prob':  result['real_prob'],
        'word_count': result['word_count'],
        'model_used': result['model_used'],
        'entry_id':   entry['id'],
        'explanation': result.get('explanation', {})
    })


@app.route('/history')
def history():
    """
    PREDICTION HISTORY PAGE
    Displays all previous predictions with details.
    Admin only access.
    """
    if not session.get('is_admin'):
        flash('Access denied. Admin only.', 'error')
        return redirect(url_for('index'))
    
    all_history = get_history()
    stats = {
        'total': len(all_history),
        'fake':  sum(1 for h in all_history if h['label'] == 'Fake'),
        'real':  sum(1 for h in all_history if h['label'] == 'Real'),
    }
    return render_template('history.html', history=all_history, stats=stats)


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clears all prediction history from shared file."""
    save_history([])
    return redirect(url_for('history'))


@app.route('/about')
def about():
    """ABOUT PAGE - Project info, tech stack, team."""
    return render_template('about.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """LOGIN PAGE - User authentication."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        # Check registered users first
        users = load_users()
        if username in users:
            stored_user = users[username]
            input_hash = hash_password(password)
            print(f"[DEBUG] Username: {username}")
            print(f"[DEBUG] Input password hash: {input_hash}")
            print(f"[DEBUG] Stored password hash: {stored_user['password']}")
            print(f"[DEBUG] Match: {input_hash == stored_user['password']}")
            
            if stored_user['password'] == input_hash:
                session['logged_in'] = True
                session['username'] = username
                session['name'] = stored_user['name']
                session['email'] = stored_user['email']
                session['is_admin'] = stored_user.get('is_admin', False)
                flash(f'Login successful! Welcome {stored_user["name"]}.', 'success')
                return redirect(url_for('index'))
            else:
                flash('Invalid username or password.', 'error')
                return render_template('login.html')
        
        # Fallback to demo credentials
        if username == 'admin' and password == 'admin123':
            session['logged_in'] = True
            session['username'] = username
            session['name'] = 'Admin'
            session['email'] = 'admin@fakenews.com'
            session['is_admin'] = True
            flash('Login successful! Welcome Admin.', 'success')
            return redirect(url_for('index'))
        elif username == 'user' and password == 'user123':
            session['logged_in'] = True
            session['username'] = username
            session['name'] = 'Demo User'
            session['email'] = 'user@fakenews.com'
            session['is_admin'] = False
            flash('Login successful! Welcome User.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """SIGNUP PAGE - User registration."""
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        
        # Validation
        if not name or not email or not username or not password:
            flash('All fields are required.', 'error')
            return render_template('signup.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('signup.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('signup.html')
        
        # Check if username already exists
        users = load_users()
        if username in users:
            flash('Username already exists. Please choose another.', 'error')
            return render_template('signup.html')
        
        # Check if email already exists
        for user in users.values():
            if user['email'] == email:
                flash('Email already registered. Please use another email.', 'error')
                return render_template('signup.html')
        
        # Create new user
        users[username] = {
            'name': name,
            'email': email,
            'password': hash_password(password),
            'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_admin': False
        }
        save_users(users)
        
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')


@app.route('/logout')
def logout():
    """LOGOUT - Clear session and redirect."""
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))


@app.route('/api/stats')
def api_stats():
    """JSON endpoint for dashboard stats (used by JS)."""
    all_history = load_history()
    return jsonify({
        'total': len(all_history),
        'fake':  sum(1 for h in all_history if h['label'] == 'Fake'),
        'real':  sum(1 for h in all_history if h['label'] == 'Real'),
        'model_name': metadata.get('model_name', 'Unknown')
    })


# =============================================================================
#  RUN THE APP
# =============================================================================

if __name__ == '__main__':
    # Create model directory if missing
    os.makedirs('model', exist_ok=True)

    # Run in debug mode during development
    # Set debug=False for production
    app.run(debug=True, host='0.0.0.0', port=5000)
