# ⚡ FakeGuard — Fake News Detection Using Machine Learning

> **Final Year Project** | Python · Flask · Scikit-learn · NLP · TF-IDF

A machine learning–powered web application that classifies news articles as **Real** or **Fake** with confidence scores, prediction history, and a clean dashboard UI.

---

## 📑 Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Problem Statement](#problem-statement)
4. [Objectives](#objectives)
5. [System Architecture](#system-architecture)
6. [Technology Stack](#technology-stack)
7. [Project Structure](#project-structure)
8. [Methodology](#methodology)
9. [Features](#features)
10. [Setup & Installation](#setup--installation)
11. [Dataset](#dataset)
12. [Advantages](#advantages)
13. [Future Scope](#future-scope)
14. [Conclusion](#conclusion)
15. [Resume Points](#resume-points)
16. [Interview Q&A](#interview-qa)
17. [Deployment Guide](#deployment-guide)

---

## Abstract

FakeGuard is a machine learning–based web application for detecting misinformation in news articles. The system employs Natural Language Processing (NLP) techniques — specifically TF-IDF vectorization — combined with Logistic Regression and Multinomial Naive Bayes classifiers trained on a large labeled dataset of fake and real news articles. The best-performing model (Logistic Regression, ~98% accuracy) is deployed via a Flask web server with a responsive UI that accepts news text, preprocesses it in real-time, and returns a verdict with a confidence score. A prediction history module allows users to track and review past analyses.

---

## Introduction

The explosion of digital media and social networking platforms has made it trivial to publish and distribute unverified information. According to MIT research, false news spreads six times faster than true news on social media. This project addresses the critical need for automated fake news detection by building a complete end-to-end machine learning pipeline — from raw text preprocessing to web-based prediction — using classical NLP and ML techniques that are well understood, explainable, and efficient.

---

## Problem Statement

Manual fact-checking cannot keep pace with the volume of content published every minute. Readers lack efficient tools to quickly assess the credibility of a news article before sharing or acting on it. A scalable, automated system is needed that can:

- Accept raw news text as input
- Preprocess and extract meaningful features
- Classify the article as Real or Fake
- Return a confidence score for transparency
- Log predictions for review

---

## Objectives

- ✅ Build a binary text classification model for fake news detection
- ✅ Apply NLP preprocessing (tokenization, stopword removal, TF-IDF)
- ✅ Train and compare Logistic Regression and Naive Bayes classifiers
- ✅ Select and persist the best model using pickle serialization
- ✅ Develop a Flask REST API serving real-time predictions
- ✅ Create a responsive, dark-themed dashboard UI
- ✅ Display confidence probability scores per prediction
- ✅ Implement a prediction history page

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER BROWSER                         │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │  index   │   │   history    │   │    about       │  │
│  │  .html   │   │   .html      │   │    .html       │  │
│  └────┬─────┘   └──────────────┘   └────────────────┘  │
│       │ POST /predict (JSON)                            │
└───────┼─────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────┐
│                   FLASK SERVER (app.py)                  │
│                                                          │
│  1. Receive news_text                                    │
│  2. preprocess_text() → clean text                       │
│  3. vectorizer.transform() → TF-IDF matrix               │
│  4. model.predict() + predict_proba() → label + conf     │
│  5. Store in session history                             │
│  6. Return JSON response                                 │
└───────┬─────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────┐
│              ML ARTIFACTS (model/)                       │
│                                                          │
│  best_model.pkl   — Logistic Regression / Naive Bayes    │
│  vectorizer.pkl   — TF-IDF Vectorizer (fitted)           │
│  metadata.pkl     — Model name, label map                │
└─────────────────────────────────────────────────────────┘
        ▲
┌───────┴─────────────────────────────────────────────────┐
│           MODEL TRAINING (model_training.py)             │
│                                                          │
│  Fake.csv + True.csv  →  pandas DataFrame                │
│  Text Preprocessing   →  clean corpus                    │
│  TF-IDF Vectorization →  feature matrix                  │
│  80/20 Train/Test     →  model evaluation                │
│  Save best model      →  model/ directory                │
└─────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | Core programming language |
| Web Framework | Flask 3.0 | REST API + HTML templating |
| ML Library | Scikit-learn | Model training & prediction |
| NLP | NLTK | Stopword removal, tokenization |
| Data Handling | Pandas, NumPy | Dataset loading & manipulation |
| Frontend | HTML5, CSS3, JS | Responsive UI |
| Serialization | Pickle | Save/load model artifacts |
| Deployment | Gunicorn, Render | Production web server |

---

## Project Structure

```
fake_news_detector/
│
├── app.py                  # Flask web application (main entry point)
├── model_training.py       # ML training pipeline
├── requirements.txt        # Python dependencies
├── Procfile                # Deployment configuration (Render/Heroku)
├── .gitignore
├── README.md
├── users.json              # User credentials storage (auto-created)
│
├── templates/              # Jinja2 HTML templates
│   ├── index.html          # Home / detection page
│   ├── history.html        # Prediction history page (admin only)
│   ├── about.html          # About / documentation page
│   ├── login.html          # User login page
│   └── signup.html         # User registration page
│
├── static/
│   ├── css/
│   │   └── style.css       # All styles (dark theme, responsive, animated)
│   └── js/
│       └── main.js         # Frontend logic (fetch API, result rendering, explanation)
│
├── model/                  # Auto-created after training
│   ├── best_model.pkl      # Trained classifier (LR or NB)
│   ├── vectorizer.pkl      # Fitted TF-IDF vectorizer
│   └── metadata.pkl        # Model name and label mapping
│
├── utils/
│   ├── __init__.py
│   └── setup.py            # Startup helper
│
└── dataset/                # Place your CSV files here
    ├── Fake.csv
    └── True.csv
```

---

## Methodology

### 1. Data Collection
- **Fake.csv**: 24,353 articles labeled as fake news
- **True.csv**: 21,417 articles labeled as real news
- Source: [Kaggle Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

### 2. Data Preprocessing
Each article goes through a 6-step pipeline:

```
Raw Text
   ↓
1. Lowercase conversion
   ↓
2. URL removal  (regex: https?://...)
   ↓
3. HTML tag removal
   ↓
4. Punctuation & special char removal
   ↓
5. Tokenization (NLTK word_tokenize)
   ↓
6. Stopword removal (NLTK English stopwords)
   ↓
Clean Token List → Rejoined as string
```

### 3. Feature Extraction — TF-IDF

**Term Frequency (TF)**: How often a word appears in a document.
```
TF(t, d) = count(t in d) / total words in d
```

**Inverse Document Frequency (IDF)**: How rare a word is across all documents.
```
IDF(t) = log(N / df(t))
```

**TF-IDF Score**:
```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

Parameters used:
- `max_features=5000` — top 5,000 most important terms
- `ngram_range=(1,2)` — unigrams + bigrams
- `min_df=2` — term must appear in ≥ 2 documents
- `sublinear_tf=True` — log(1 + TF) to reduce impact of very common terms

### 4. Model Training

**Logistic Regression**
- Excellent for binary classification
- Outputs calibrated probabilities via sigmoid function
- `max_iter=1000, C=1.0, solver='lbfgs'`

**Multinomial Naive Bayes**
- Based on Bayes' theorem, assumes feature independence
- Works well with TF-IDF count-based features
- `alpha=0.1` (Laplace smoothing)

### 5. Evaluation Metrics
- **Accuracy** — overall correct predictions
- **Precision** — of predicted fakes, how many were actually fake
- **Recall** — of actual fakes, how many were caught
- **F1-Score** — harmonic mean of precision and recall

### 6. Model Selection
The model with the highest accuracy on the 20% test set is automatically saved.

---

## Features

- 🔍 **Real-time detection** — results in milliseconds
- 📊 **Confidence scores** — fake% vs real% probability bars
- 🧠 **Explainable AI** — shows why news is classified as fake/real with key word analysis
- � **User authentication** — signup/login system with password hashing
- 🔒 **Admin-only history** — prediction history restricted to admin users
- �� **Prediction history** — session-based log of all analyses (admin access)
- 🧹 **Sample texts** — one-click fake/real example injection
- 📱 **Responsive UI** — works on mobile, tablet, desktop
- 🎨 **Animated background** — moving light rays and floating dots
- 🤖 **Model info** — displays which model made the prediction
- ⌨️ **Keyboard shortcut** — Ctrl+Enter to analyse
- 📈 **Live stats dashboard** — running totals on the home page

---

## Authentication System

The application includes a complete user authentication system:

### User Roles
- **Admin**: Full access including prediction history page
  - Demo credentials: `admin` / `admin123`
- **Regular User**: Basic access to news detection only
  - Demo credentials: `user` / `user123`

### User Registration
- Users can sign up with: Full Name, Email, Username, Password, Confirm Password
- Passwords are hashed using SHA-256 before storage
- Validation includes: required fields, minimum password length (6 chars), password matching, unique username, unique email
- User data stored in `users.json` file

### Login Process
- Checks registered users first (hashed password comparison)
- Falls back to demo credentials if no match found
- Session stores: username, name, email, is_admin flag
- Admin users can access `/history` route
- Regular users redirected with access denied message

---

## Setup & Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

### Step 2 — Create and activate virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Add the dataset
Create a `dataset/` folder and place your CSV files inside:
```
dataset/
├── Fake.csv
└── True.csv
```
> If you don't have the dataset, the app will auto-generate a synthetic demo dataset.

### Step 5 — Train the model
```bash
python model_training.py
```
This will:
- Load and preprocess the data
- Train Logistic Regression and Naive Bayes
- Compare accuracy
- Save the best model to `model/`

### Step 6 — Run the Flask app
```bash
python app.py
```
Open your browser at: **http://localhost:5000**

---

## Dataset

Download from Kaggle:
**https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset**

| File | Records | Label |
|---|---|---|
| Fake.csv | 24,353 | 0 (Fake) |
| True.csv | 21,417 | 1 (Real) |

Columns used: `title`, `text` (concatenated)

---

## Advantages

- ⚡ Real-time predictions with no external API dependency
- 🔒 Runs fully offline — no data sent to third parties
- 📊 Transparent confidence scores build user trust
- 🧩 Modular codebase — easy to swap in BERT or other models
- 💡 Explainable ML — Logistic Regression is interpretable
- 📱 Fully responsive — accessible on any device
- 🗂 Session-based history — track all checks in one session

---

## Future Scope

- 🧠 Integrate transformer models (BERT, RoBERTa) for higher accuracy
- 🌐 Add URL scanning — fetch and analyse articles from links
- 🌍 Multi-language support using multilingual NLP models
- 🔌 Build a Chrome extension for inline news verification
- 📡 Real-time dataset pipeline — retrain on new articles periodically
- 📊 Admin dashboard with full analytics (charts, exports)
- 🔗 Social media API integration (Twitter, Reddit feed scanning)
- 🗄 PostgreSQL / MongoDB for persistent history (replace sessions)

---

## Conclusion

FakeGuard demonstrates that classical NLP + ML pipelines remain highly effective for fake news detection. Logistic Regression with TF-IDF achieves approximately 98% accuracy on the test set — outperforming many complex deep learning approaches on the same dataset while being orders of magnitude faster to train and deploy. The project provides a complete, production-ready template that can be extended with advanced models, persistent storage, and richer feature sets as requirements grow.

---

## Resume Points

Use these 5 bullet points on your resume under Projects:

```
• Developed a full-stack Fake News Detection web application using Flask and
  Scikit-learn; trained Logistic Regression and Naive Bayes classifiers on a
  45,000-article dataset, achieving 98%+ classification accuracy.

• Built an end-to-end NLP preprocessing pipeline (lowercasing, URL removal,
  stopword filtering, tokenization) and TF-IDF vectorization with 5,000
  features and bigram support, reducing noise and improving model performance.

• Designed a REST API in Flask that accepts raw news text, preprocesses it,
  runs ML inference, and returns real-time predictions with fake/real
  confidence probability scores.

• Created a responsive dark-themed dashboard (HTML, CSS, Vanilla JS) with
  dynamic confidence bars, live stats counters, session-based prediction
  history, and keyboard shortcuts — achieving a seamless UX across devices.

• Deployed the application on Render using Gunicorn; managed ML model
  serialization with Pickle and implemented session-based prediction logging,
  demonstrating full production-readiness of the system.
```

---

## Interview Q&A

### 1. What is fake news detection and why is it important?
**A:** Fake news detection is the task of automatically classifying news articles as real or fabricated. It's important because misinformation spreads rapidly on social media, influencing public opinion, elections, health decisions, and social stability. Manual fact-checking cannot scale, so automated ML-based systems are essential.

---

### 2. What dataset did you use and how was it structured?
**A:** I used the Kaggle Fake and Real News Dataset consisting of two CSV files: `Fake.csv` (24,353 fake articles) and `True.csv` (21,417 real articles). Each file has columns for `title`, `text`, `subject`, and `date`. I concatenated title + text to form a single `content` field, labeled fake as 0 and real as 1, then merged both into a single DataFrame.

---

### 3. Explain TF-IDF and why you used it.
**A:** TF-IDF stands for Term Frequency–Inverse Document Frequency. TF measures how often a word appears in a document; IDF measures how rare that word is across the entire corpus. Multiplying them gives high weight to words that are frequent in a document but rare globally — these are the most distinguishing words. I used it because it converts raw text into numerical feature vectors that ML models can process, and it naturally suppresses common but uninformative words like "the" or "is".

---

### 4. What preprocessing steps did you apply?
**A:** Six steps: (1) convert to lowercase, (2) remove URLs using regex, (3) strip HTML tags, (4) remove punctuation and special characters, (5) tokenize using NLTK's `word_tokenize`, and (6) remove English stopwords. This cleans the text and ensures the vectorizer focuses on meaningful content words.

---

### 5. Why did you choose Logistic Regression over Naive Bayes?
**A:** Both were trained and evaluated. Logistic Regression generally outperforms Naive Bayes on TF-IDF features because it doesn't assume feature independence (Naive Bayes' key assumption). In practice, word co-occurrences are correlated, and LR handles this better. On our dataset, LR achieved ~98% accuracy vs ~95% for NB. The script automatically selects the better model.

---

### 6. How does Logistic Regression classify text?
**A:** LR applies a linear combination of input features (TF-IDF weights) plus a bias term, then passes the result through a sigmoid function: `σ(z) = 1 / (1 + e^(-z))`. This maps the output to a probability between 0 and 1. If probability > 0.5, the article is classified as Real; otherwise Fake. The model learns optimal feature weights during training by minimizing log-loss.

---

### 7. What is the train-test split and why 80/20?
**A:** We split the dataset so 80% is used for training the model and 20% for evaluation. The test set simulates unseen data and gives an unbiased estimate of real-world performance. 80/20 is the standard split — enough training data for the model to learn, and enough test data for reliable evaluation. We use `stratify=y` to ensure both splits have equal proportions of fake and real articles.

---

### 8. How do you save and load the trained model?
**A:** Using Python's `pickle` module. After training, `pickle.dump(model, f)` serializes the model object to a `.pkl` binary file. At Flask startup, `pickle.load(f)` deserializes it back into memory. The same is done for the TF-IDF vectorizer, which must be saved separately because it stores the vocabulary learned during `fit_transform`.

---

### 9. Why must the same vectorizer be used for training and prediction?
**A:** The vectorizer learns a fixed vocabulary during `fit_transform()` on the training set — mapping each word to a column index. If you use a different vectorizer at prediction time, the feature dimensions won't match, and the model will receive garbled input. Using `transform()` (not `fit_transform()`) on new text ensures the same vocabulary mapping is applied.

---

### 10. What is Flask and how does it serve the ML model?
**A:** Flask is a lightweight Python web framework. In `app.py`, I define route handlers using decorators like `@app.route('/predict', methods=['POST'])`. When the browser sends a POST request with news text, Flask calls the handler, which runs the preprocessing + inference pipeline and returns a JSON response. The HTML templates are rendered using Jinja2, Flask's built-in templating engine.

---

### 11. How do you display the confidence score?
**A:** Scikit-learn's `predict_proba()` method returns a 2D array where each row has `[P(Fake), P(Real)]`. I multiply by 100 to get percentages, then return them in the JSON response. The frontend uses these values to animate the width of CSS-based progress bars, creating a visual confidence indicator.

---

### 12. What is an ngram and why did you use bigrams?
**A:** An n-gram is a contiguous sequence of n words. A unigram is a single word; a bigram is two consecutive words. With `ngram_range=(1,2)`, the vectorizer captures both individual words and two-word phrases. Bigrams preserve context — e.g., "not guilty" means something very different from "not" and "guilty" separately — which improves classification accuracy.

---

### 13. How does the prediction history feature work?
**A:** Flask's `session` object (a signed cookie) stores a list of prediction dictionaries including the text snippet, label, confidence, probabilities, word count, and timestamp. After each prediction, the new entry is prepended to the list (newest first), and the list is trimmed to 50 entries. The `/history` route reads from the session and renders the history template.

---

### 14. What are precision and recall and why do both matter?
**A:** Precision is "of all articles predicted as Fake, how many were actually Fake" — it measures false alarm rate. Recall is "of all genuinely Fake articles, how many did we catch" — it measures miss rate. In fake news detection, high recall is critical (we don't want to miss real fake news), but we also want high precision to avoid flagging legitimate articles. F1-score balances both.

---

### 15. What is the confusion matrix?
**A:** A 2×2 table showing True Positives (correctly predicted Fake), True Negatives (correctly predicted Real), False Positives (real news wrongly called Fake), and False Negatives (fake news missed). It gives a fuller picture of model performance than accuracy alone, especially on imbalanced datasets.

---

### 16. How would you improve accuracy further?
**A:** Several ways: (1) Use transformer models like BERT or RoBERTa which understand semantic context. (2) Add source-level features (URL domain reputation). (3) Use ensemble methods (Random Forest, Gradient Boosting). (4) Include metadata features like article length, publication date, author. (5) Use word embeddings (Word2Vec, GloVe) instead of TF-IDF.

---

### 17. What are stopwords and why remove them?
**A:** Stopwords are common words like "the", "is", "at", "which" that appear in virtually every document and carry no discriminating information for classification. Removing them reduces feature space, speeds up training, and prevents these words from diluting the importance of meaningful, class-specific terms. NLTK provides a standard English stopwords list of ~179 words.

---

### 18. Can your model detect fake news in languages other than English?
**A:** Currently no — the TF-IDF vectorizer and NLTK stopwords are configured for English, and the training dataset is English-only. To support other languages, you would need translated or language-specific datasets, a multilingual tokenizer (like spaCy's multi-language models), and language-specific stopword lists. Multilingual transformer models like mBERT handle this naturally.

---

### 19. What is Gunicorn and why is it used for deployment?
**A:** Gunicorn (Green Unicorn) is a production-grade WSGI HTTP server for Python web apps. Flask's built-in dev server is single-threaded and not safe for production traffic. Gunicorn runs multiple worker processes, handles concurrent requests, and integrates with Render's infrastructure. The `Procfile` tells Render to start the app with `gunicorn app:app`.

---

### 20. What challenges did you face in this project?
**A:** Key challenges: (1) **Data imbalance** — fake and real datasets were slightly unequal; solved with stratified splitting. (2) **Preprocessing consistency** — the exact same preprocessing must be applied at training and inference; solved by sharing the `preprocess_text()` function. (3) **Model persistence** — both model and vectorizer must be saved/loaded correctly; solved with careful pickle serialization. (4) **Session storage limits** — browser cookies cap at 4KB; solved by trimming history to 50 entries and storing only text snippets (200 chars max).

---

## Deployment Guide

### Deploy on GitHub

```bash
# 1. Initialize git repository
cd fake_news_detector
git init

# 2. Add all files
git add .

# 3. Commit
git commit -m "Initial commit: FakeGuard fake news detection app"

# 4. Create a new repo on GitHub, then push
git remote add origin https://github.com/yourusername/fake-news-detector.git
git branch -M main
git push -u origin main
```

> **Important**: Before pushing, run `python model_training.py` and commit the `model/` folder, OR add a `build` command in Render to auto-train.

---

### Deploy on Render (Free Tier)

1. **Push your code to GitHub** (see above)

2. **Go to** [render.com](https://render.com) and sign up / log in

3. **New → Web Service** → Connect your GitHub repo

4. **Configure the service:**

| Setting | Value |
|---|---|
| **Name** | fake-news-detector |
| **Environment** | Python 3 |
| **Build Command** | `pip install -r requirements.txt && python model_training.py` |
| **Start Command** | `gunicorn app:app` |
| **Plan** | Free |

5. Click **Deploy**

6. Render will:
   - Install dependencies
   - Train the ML model
   - Start the Gunicorn server
   - Give you a public URL: `https://fake-news-detector.onrender.com`

---

### Environment Variables (Optional)

Set these in Render's Environment tab if needed:

```
FLASK_ENV=production
SECRET_KEY=your_random_secret_key_here
```

---

*Built with ❤️ using Python, Flask, and Scikit-learn*
