import pickle

# Load model and vectorizer
with open('model/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('model/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print("Metadata:", metadata)
print()

# Test with fake news samples
fake_news = [
    "SHOCKING: Scientists discover that drinking bleach cures cancer overnight!",
    "BREAKING: The moon is made of cheese, NASA confirms after secret mission",
    "5G towers are spreading coronavirus! Government covering up the truth!",
]

# Test with real news samples
real_news = [
    "Federal Reserve raises interest rates by 0.25% to combat rising inflation",
    "WHO reports progress in malaria vaccine trials across sub-Saharan Africa",
    "Senate passes bipartisan infrastructure bill worth 1.2 trillion dollars",
]

print("Testing FAKE news samples:")
for i, text in enumerate(fake_news, 1):
    clean_text = text.lower()
    text_tfidf = vectorizer.transform([clean_text])
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    label = "Real" if prediction == 1 else "Fake"
    print(f"{i}. {text[:50]}...")
    print(f"   Prediction: {label} (prob: Fake={probabilities[0]:.2f}, Real={probabilities[1]:.2f})")
    print()

print("Testing REAL news samples:")
for i, text in enumerate(real_news, 1):
    clean_text = text.lower()
    text_tfidf = vectorizer.transform([clean_text])
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    label = "Real" if prediction == 1 else "Fake"
    print(f"{i}. {text[:50]}...")
    print(f"   Prediction: {label} (prob: Fake={probabilities[0]:.2f}, Real={probabilities[1]:.2f})")
    print()
