"""
=============================================================================
  FAKE NEWS DETECTION - MODEL TRAINING SCRIPT
  File: model_training.py
  Author: [Your Name]
  Description:
      This script loads the Fake.csv and True.csv datasets, preprocesses the
      text data, trains Logistic Regression and Naive Bayes models using
      TF-IDF vectorization, compares accuracy, and saves the best model.
=============================================================================
"""

import os
import re
import pickle
import numpy as np
import pandas as pd

# --- NLP & ML Libraries ---
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

# Download required NLTK data (runs once)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# =============================================================================
#  STEP 1: LOAD DATASET
# =============================================================================

def load_dataset(fake_path="dataset/Fake.csv", true_path="dataset/True.csv"):
    """
    Loads Fake.csv and True.csv, assigns labels, and merges them.
    Label: 0 = Fake News, 1 = Real News
    """
    print("[INFO] Loading datasets...")

    # Check if dataset files exist
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        print("[WARNING] Dataset files not found. Using synthetic dataset for demo.")
        return generate_synthetic_dataset()

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Assign labels
    fake_df['label'] = 0  # 0 = Fake
    true_df['label'] = 1  # 1 = Real

    # Combine title + text for richer features
    fake_df['content'] = fake_df.get('title', '') + " " + fake_df.get('text', '')
    true_df['content'] = true_df.get('title', '') + " " + true_df.get('text', '')

    # Merge datasets
    df = pd.concat([fake_df[['content', 'label']],
                    true_df[['content', 'label']]], ignore_index=True)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[INFO] Dataset loaded: {len(df)} articles "
          f"({len(fake_df)} fake, {len(true_df)} real)")
    return df


def generate_synthetic_dataset():
    """
    Generates a small synthetic dataset when CSV files are not present.
    This allows the app to run in demo mode without the actual dataset.
    """
    print("[INFO] Generating synthetic demo dataset...")

    fake_samples = [
        "SHOCKING: Scientists discover that drinking bleach cures cancer overnight!",
        "BREAKING: The moon is made of cheese, NASA confirms after secret mission",
        "You won't believe what Obama did in secret! Elites are hiding this from you!",
        "5G towers are spreading coronavirus! Government covering up the truth!",
        "EXCLUSIVE: Vaccines contain microchips to track every citizen worldwide",
        "Aliens have landed in Texas! Government orders media blackout immediately!",
        "Scientists prove Earth is flat! NASA has been lying for 60 years!",
        "Miracle cure discovered: eat garlic and lemon to eliminate all diseases!",
        "LEAKED: Secret documents reveal the moon landing was staged in Hollywood!",
        "Bill Gates wants to reduce world population using vaccines, insiders reveal!",
        "Doctors don't want you to know this one weird trick to cure diabetes!",
        "Breaking: President secretly funds terrorist organizations abroad!",
        "Scientists baffled: Man grows third arm after eating GM foods daily!",
        "SHOCKING TRUTH: Water fluoridation is government mind control experiment!",
        "Celebrity reveals secret society controlling all world governments today!",
        "WARNING: New phone update installs spyware directly into your brain!",
        "EXPOSED: Big pharma hiding cheap cancer cure to maximize profits!",
        "Earthquake weapon used by US military to cause natural disasters worldwide!",
        "Secret treaty gives China control over all US military bases revealed!",
        "Scientists confirm: Chemtrails contain mind-altering chemicals from planes!",
    ] * 15  # Repeat to get more samples

    real_samples = [
        "Federal Reserve raises interest rates by 0.25% to combat rising inflation",
        "WHO reports progress in malaria vaccine trials across sub-Saharan Africa",
        "Senate passes bipartisan infrastructure bill worth 1.2 trillion dollars",
        "NASA's James Webb telescope captures new images of distant galaxies",
        "Stock markets decline as investors react to new inflation data reports",
        "Scientists publish peer-reviewed study on climate change in Nature journal",
        "Olympic committee announces host city for 2032 Summer Olympic Games",
        "Central bank releases quarterly economic outlook report for next year",
        "University researchers develop new battery technology for electric vehicles",
        "World Health Organization issues updated guidelines on COVID-19 treatment",
        "Supreme Court issues ruling on landmark environmental protection case",
        "International trade negotiations conclude with new tariff agreements",
        "Researchers at MIT publish breakthrough in quantum computing technology",
        "Government announces new education funding package for rural schools",
        "Tech companies agree to new data privacy standards set by regulators",
        "Medical researchers identify new genetic markers linked to heart disease",
        "United Nations Security Council votes on new peacekeeping resolution",
        "Economists predict moderate GDP growth for the coming fiscal quarter",
        "New carbon capture technology achieves record efficiency in pilot test",
        "Pharmaceutical company completes Phase 3 trials for new diabetes drug",
    ] * 15  # Repeat to get more samples

    fake_df = pd.DataFrame({'content': fake_samples, 'label': [0] * len(fake_samples)})
    real_df = pd.DataFrame({'content': real_samples, 'label': [1] * len(real_samples)})

    df = pd.concat([fake_df, real_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[INFO] Synthetic dataset created: {len(df)} samples")
    return df


# =============================================================================
#  STEP 2: TEXT PREPROCESSING
# =============================================================================

def preprocess_text(text):
    """
    Cleans and preprocesses raw news text:
      1. Convert to lowercase
      2. Remove URLs
      3. Remove punctuation and special characters
      4. Tokenize
      5. Remove stopwords
      6. Rejoin tokens
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # 4. Remove punctuation and special characters (keep letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)

    # 5. Tokenize
    tokens = word_tokenize(text)

    # 6. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # 7. Rejoin
    return ' '.join(tokens)


# =============================================================================
#  STEP 3: FEATURE EXTRACTION (TF-IDF)
# =============================================================================

def extract_features(X_train, X_test, max_features=5000):
    """
    Converts text to numerical features using TF-IDF Vectorization.
    TF-IDF = Term Frequency - Inverse Document Frequency
    Gives higher weight to words that are important in a document
    but rare across all documents.
    """
    print("[INFO] Extracting TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=max_features,   # Keep top 5000 words
        ngram_range=(1, 2),          # Use unigrams and bigrams
        min_df=2,                    # Word must appear in at least 2 docs
        sublinear_tf=True            # Apply log normalization
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    print(f"[INFO] Feature matrix shape: {X_train_tfidf.shape}")
    return vectorizer, X_train_tfidf, X_test_tfidf


# =============================================================================
#  STEP 4: TRAIN MODELS
# =============================================================================

def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression classifier.
    LR is excellent for binary classification tasks like fake/real.
    """
    print("[INFO] Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(X_train, y_train):
    """
    Trains a Multinomial Naive Bayes classifier.
    NB works well with TF-IDF features and text classification.
    """
    print("[INFO] Training Naive Bayes...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)
    return model


# =============================================================================
#  STEP 5: EVALUATE & SELECT BEST MODEL
# =============================================================================

def evaluate_model(model, X_test, y_test, name):
    """
    Evaluates a model and prints classification report.
    Returns accuracy score.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  Model: {name}")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred,
                                target_names=['Fake', 'Real']))
    return accuracy


# =============================================================================
#  STEP 6: SAVE MODEL & VECTORIZER
# =============================================================================

def save_artifacts(model, vectorizer, model_name):
    """
    Saves the trained model and TF-IDF vectorizer to disk using pickle.
    These are loaded later by the Flask app for predictions.
    """
    os.makedirs('model', exist_ok=True)

    with open('model/best_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # Save model metadata
    metadata = {
        'model_name': model_name,
        'labels': {0: 'Fake', 1: 'Real'}
    }
    with open('model/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\n[INFO] ✅ Model saved: model/best_model.pkl")
    print(f"[INFO] ✅ Vectorizer saved: model/vectorizer.pkl")


# =============================================================================
#  MAIN TRAINING PIPELINE
# =============================================================================

def main():
    print("\n" + "="*60)
    print("   FAKE NEWS DETECTION - MODEL TRAINING PIPELINE")
    print("="*60 + "\n")

    # Step 1: Load data
    df = load_dataset()

    # Step 2: Preprocess text
    print("\n[INFO] Preprocessing text data (this may take a moment)...")
    df['clean_content'] = df['content'].apply(preprocess_text)

    # Remove empty rows after preprocessing
    df = df[df['clean_content'].str.strip() != '']
    print(f"[INFO] Clean dataset size: {len(df)} samples")

    # Step 3: Split dataset
    X = df['clean_content']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    # Step 4: TF-IDF Feature Extraction
    vectorizer, X_train_tfidf, X_test_tfidf = extract_features(X_train, X_test)

    # Step 5: Train both models
    lr_model = train_logistic_regression(X_train_tfidf, y_train)
    nb_model = train_naive_bayes(X_train_tfidf, y_train)

    # Step 6: Evaluate both models
    print("\n[INFO] Evaluating models...")
    lr_accuracy = evaluate_model(lr_model, X_test_tfidf, y_test, "Logistic Regression")
    nb_accuracy = evaluate_model(nb_model, X_test_tfidf, y_test, "Naive Bayes")

    # Step 7: Select best model
    if lr_accuracy >= nb_accuracy:
        best_model = lr_model
        best_name  = "Logistic Regression"
        best_acc   = lr_accuracy
    else:
        best_model = nb_model
        best_name  = "Naive Bayes"
        best_acc   = nb_accuracy

    print(f"\n[RESULT] 🏆 Best Model: {best_name} (Accuracy: {best_acc*100:.2f}%)")

    # Step 8: Save best model
    save_artifacts(best_model, vectorizer, best_name)

    print("\n[INFO] Training complete! Run 'python app.py' to start the web app.")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
