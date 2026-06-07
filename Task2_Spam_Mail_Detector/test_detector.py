"""
📧 Spam Detector — Interactive Test Mode
Test the trained model with your own messages!
"""
import pickle
import os
import re
import string
import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import urllib.request
import zipfile
import io

# ── Setup ──
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

# ── Load & Train ──
print("\n  Loading dataset & training model... ", end="", flush=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
local_path = os.path.join(script_dir, "SMSSpamCollection")

if os.path.exists(local_path):
    df = pd.read_csv(local_path, sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
else:
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    response = urllib.request.urlopen(data_url, timeout=30)
    zip_data = io.BytesIO(response.read())
    with zipfile.ZipFile(zip_data, 'r') as z:
        for name in z.namelist():
            if 'SMSSpamCollection' in name and not name.endswith('/'):
                with z.open(name) as f:
                    df = pd.read_csv(f, sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
                z.extract(name, script_dir)
                break

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])
df['cleaned'] = df['message'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['label_encoded'], test_size=0.2, random_state=42, stratify=df['label_encoded']
)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LinearSVC(max_iter=2000, random_state=42, C=1.0)
model.fit(X_train_tfidf, y_train)

acc = accuracy_score(y_test, model.predict(X_test_tfidf))
print(f"Done! (Accuracy: {acc*100:.1f}%)")

# ── Interactive Loop ──
print()
print("=" * 60)
print("   SPAM DETECTOR - Interactive Test Mode")
print("=" * 60)
print()
print("  Type any message and press Enter to classify it.")
print("  Type 'quit' or 'exit' to stop.")
print("  Type 'examples' to see sample test messages.")
print()

examples = [
    ("Congratulations! You've won a $1000 gift card. Call now to claim!", "spam"),
    ("Hey, are you coming to the party tonight?", "ham"),
    ("URGENT: Your account has been compromised. Click here to verify.", "spam"),
    ("Can you pick up some groceries on the way home?", "ham"),
    ("FREE entry to win a brand new iPhone! Text WIN to 80085", "spam"),
    ("Thanks for lunch today, it was really nice catching up!", "ham"),
    ("You have been selected for a cash prize of $5000! Reply YES", "spam"),
    ("Don't forget we have a meeting at 3pm tomorrow", "ham"),
    ("Act now! Limited time offer - 90% off all products!", "spam"),
    ("I'll be there in 10 minutes, wait for me", "ham"),
]

while True:
    try:
        msg = input("  >> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\n  Goodbye!\n")
        break

    if not msg:
        continue
    if msg.lower() in ('quit', 'exit', 'q'):
        print("\n  Goodbye!\n")
        break
    if msg.lower() == 'examples':
        print()
        print(f"  {'Message':<62} {'Expected':>8}")
        print(f"  {'-'*72}")
        for ex_msg, ex_label in examples:
            short = ex_msg[:58] + "..." if len(ex_msg) > 58 else ex_msg
            print(f"  {short:<62} {ex_label:>8}")
        print()
        print("  Copy-paste any message above to test it!")
        print()
        continue

    # Classify
    cleaned = clean_text(msg)
    features = tfidf.transform([cleaned])
    prediction = model.predict(features)[0]
    decision = model.decision_function(features)[0]
    confidence = min(abs(decision) / 2 * 100, 99.9)  # rough confidence

    if prediction == 1:
        print(f"  >>> SPAM  (confidence: {confidence:.1f}%)")
    else:
        print(f"  >>> HAM   (confidence: {confidence:.1f}%)")
    print()
