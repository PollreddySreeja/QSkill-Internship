"""
📧 Spam Mail Detector — Web Application
Flask backend serving the trained ML model with a premium UI.
"""
import os
import re
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from flask import Flask, render_template, request, jsonify, send_from_directory
import urllib.request
import zipfile
import io

# ═══════════════════════════════════════════════════════════
# GLOBALS
# ═══════════════════════════════════════════════════════════
app = Flask(__name__, static_folder='static', template_folder='templates')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# These will be populated on startup
model = None
tfidf = None
all_results = {}
dataset_info = {}


def clean_text(text):
    """NLP preprocessing pipeline."""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)


def load_and_train():
    """Load dataset, train all models, return best."""
    global model, tfidf, all_results, dataset_info

    print("  [1/4] Loading dataset...")
    local_path = os.path.join(SCRIPT_DIR, "SMSSpamCollection")

    if os.path.exists(local_path):
        df = pd.read_csv(local_path, sep='\t', header=None,
                         names=['label', 'message'], encoding='latin-1')
    else:
        data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        response = urllib.request.urlopen(data_url, timeout=30)
        zip_data = io.BytesIO(response.read())
        with zipfile.ZipFile(zip_data, 'r') as z:
            for name in z.namelist():
                if 'SMSSpamCollection' in name and not name.endswith('/'):
                    with z.open(name) as f:
                        df = pd.read_csv(f, sep='\t', header=None,
                                         names=['label', 'message'], encoding='latin-1')
                    z.extract(name, SCRIPT_DIR)
                    break

    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])

    dataset_info = {
        'total': len(df),
        'ham': int((df['label'] == 'ham').sum()),
        'spam': int((df['label'] == 'spam').sum()),
        'ham_pct': round((df['label'] == 'ham').mean() * 100, 1),
        'spam_pct': round((df['label'] == 'spam').mean() * 100, 1),
    }

    print("  [2/4] Preprocessing text...")
    df['cleaned'] = df['message'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned'], df['label_encoded'],
        test_size=0.2, random_state=42, stratify=df['label_encoded']
    )

    print("  [3/4] Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2),
        min_df=2, max_df=0.95, sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("  [4/4] Training models...")
    models = {
        'Multinomial Naive Bayes': MultinomialNB(alpha=1.0),
        'Complement Naive Bayes': ComplementNB(alpha=1.0),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        'Linear SVM': LinearSVC(max_iter=2000, random_state=42, C=1.0),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, random_state=42, max_depth=5),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, m in models.items():
        cv_scores = cross_val_score(m, X_train_tfidf, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        m.fit(X_train_tfidf, y_train)
        y_pred = m.predict(X_test_tfidf)

        all_results[name] = {
            'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
            'precision': round(precision_score(y_test, y_pred) * 100, 2),
            'recall': round(recall_score(y_test, y_pred) * 100, 2),
            'f1': round(f1_score(y_test, y_pred) * 100, 2),
            'cv_mean': round(cv_scores.mean() * 100, 2),
            'cv_std': round(cv_scores.std() * 100, 2),
        }

    # Best model by F1
    best_name = max(all_results, key=lambda k: all_results[k]['f1'])
    model = models[best_name]

    print(f"\n  ✅ Ready! Best model: {best_name} (F1: {all_results[best_name]['f1']}%)")
    return best_name


# ═══════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '').strip()

    if not message:
        return jsonify({'error': 'Empty message'}), 400

    cleaned = clean_text(message)
    features = tfidf.transform([cleaned])
    prediction = model.predict(features)[0]

    # Get confidence from decision function
    if hasattr(model, 'decision_function'):
        decision = model.decision_function(features)[0]
        confidence = min(abs(decision) / 2 * 100, 99.9)
    elif hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features)[0]
        confidence = round(max(proba) * 100, 1)
    else:
        confidence = 95.0

    label = 'spam' if prediction == 1 else 'ham'

    return jsonify({
        'label': label,
        'confidence': round(confidence, 1),
        'cleaned_text': cleaned,
        'original': message,
    })


@app.route('/api/stats')
def stats():
    return jsonify({
        'dataset': dataset_info,
        'models': all_results,
    })


@app.route('/visualizations/<path:filename>')
def serve_viz(filename):
    return send_from_directory(SCRIPT_DIR, filename)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("   SPAM MAIL DETECTOR — Web Application")
    print("=" * 55 + "\n")

    best = load_and_train()

    print(f"\n  Starting web server...")
    print(f"  Open in browser: http://127.0.0.1:5000")
    print(f"  Press Ctrl+C to stop.\n")

    app.run(debug=False, host='127.0.0.1', port=5000)
