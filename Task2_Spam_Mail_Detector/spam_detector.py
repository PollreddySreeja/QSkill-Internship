"""
╔══════════════════════════════════════════════════════════════════════╗
║                    📧  SPAM MAIL DETECTOR  📧                       ║
║              QSkill AI & ML Internship — Task 2                     ║
║                                                                      ║
║  Build a classifier that distinguishes between spam and non-spam     ║
║  (ham) emails using textual data with NLP preprocessing.             ║
║                                                                      ║
║  Author : Pollreddy Sreeja                                           ║
║  Date   : June 2026                                                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════
import os
import re
import string
import warnings
import urllib.request
import zipfile
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns

from collections import Counter

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ML
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, learning_curve
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Word Cloud
from wordcloud import WordCloud

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# GLOBAL STYLING
# ═══════════════════════════════════════════════════════════════════════

# Premium dark theme color palette (unique to Task 2)
PALETTE = {
    'bg_dark':       '#0F0F1A',
    'bg_card':       '#1A1A2E',
    'accent_cyan':   '#00D4FF',
    'accent_magenta':'#FF2E63',
    'accent_gold':   '#FFD700',
    'accent_green':  '#00E676',
    'accent_purple': '#B388FF',
    'text_light':    '#E8E8F0',
    'text_muted':    '#8888AA',
    'grid':          '#2A2A44',
    'ham_color':     '#00E676',
    'spam_color':    '#FF2E63',
}

def setup_plot_style():
    """Configure matplotlib with a premium dark aesthetic."""
    plt.rcParams.update({
        'figure.facecolor':   PALETTE['bg_dark'],
        'axes.facecolor':     PALETTE['bg_card'],
        'axes.edgecolor':     PALETTE['grid'],
        'axes.labelcolor':    PALETTE['text_light'],
        'axes.titlesize':     15,
        'axes.titleweight':   'bold',
        'axes.labelsize':     12,
        'xtick.color':        PALETTE['text_muted'],
        'ytick.color':        PALETTE['text_muted'],
        'text.color':         PALETTE['text_light'],
        'figure.figsize':     (14, 8),
        'figure.dpi':         120,
        'savefig.dpi':        150,
        'savefig.facecolor':  PALETTE['bg_dark'],
        'savefig.edgecolor':  PALETTE['bg_dark'],
        'font.family':        'sans-serif',
        'font.size':          11,
        'grid.color':         PALETTE['grid'],
        'grid.alpha':         0.4,
        'legend.facecolor':   PALETTE['bg_card'],
        'legend.edgecolor':   PALETTE['grid'],
        'legend.fontsize':    10,
    })


# ═══════════════════════════════════════════════════════════════════════
# CONSOLE OUTPUT HELPERS
# ═══════════════════════════════════════════════════════════════════════

def banner(text, width=68):
    """Print a bold section banner."""
    print()
    print("╔" + "═" * width + "╗")
    print("║" + text.center(width) + "║")
    print("╚" + "═" * width + "╝")

def section(text):
    """Print a sub-section header."""
    print(f"\n{'─' * 60}")
    print(f"  ▸ {text}")
    print(f"{'─' * 60}")

def kv(key, value, indent=4):
    """Print a key-value pair."""
    spaces = " " * indent
    print(f"{spaces}• {key:<28} {value}")

def progress(step, total, label=""):
    """Print a simple progress indicator."""
    pct = step / total * 100
    bar_len = 30
    filled = int(bar_len * step / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"    [{bar}] {pct:5.1f}%  {label}", end="\r" if step < total else "\n")


# ═══════════════════════════════════════════════════════════════════════
# CLASS: SpamDetectorPipeline
# ═══════════════════════════════════════════════════════════════════════

class SpamDetectorPipeline:
    """
    End-to-end Spam Mail Detection pipeline.
    
    Workflow:
        1. Load dataset (SMS Spam Collection from UCI)
        2. Explore & visualize data
        3. Preprocess text (clean, tokenize, remove stopwords, stem)
        4. Extract features (TF-IDF)
        5. Train & compare models
        6. Evaluate best model with detailed metrics
        7. Generate professional visualizations
    """

    def __init__(self, output_dir=None):
        self.output_dir = output_dir or os.path.dirname(os.path.abspath(__file__))
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.tfidf = None
        self.X_train_tfidf = None
        self.X_test_tfidf = None
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        self.label_encoder = LabelEncoder()
        self.stemmer = PorterStemmer()

        # Download NLTK data silently
        for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
            nltk.download(resource, quiet=True)

        self.stop_words = set(stopwords.words('english'))
        setup_plot_style()

    # ───────────────────────────────────────────────────────────────
    # STEP 1: Load Dataset
    # ───────────────────────────────────────────────────────────────

    def load_data(self):
        """Load the SMS Spam Collection dataset from UCI repository."""
        banner("STEP 1 ─ LOAD DATASET")

        data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        local_path = os.path.join(self.output_dir, "SMSSpamCollection")

        if os.path.exists(local_path):
            print("    📂 Found local dataset file.")
            self.df = pd.read_csv(
                local_path, sep='\t', header=None,
                names=['label', 'message'], encoding='latin-1'
            )
        else:
            print("    🌐 Downloading SMS Spam Collection from UCI...")
            try:
                response = urllib.request.urlopen(data_url, timeout=30)
                zip_data = io.BytesIO(response.read())
                with zipfile.ZipFile(zip_data, 'r') as z:
                    # The file inside the zip
                    for name in z.namelist():
                        if 'SMSSpamCollection' in name and not name.endswith('/'):
                            with z.open(name) as f:
                                self.df = pd.read_csv(
                                    f, sep='\t', header=None,
                                    names=['label', 'message'], encoding='latin-1'
                                )
                            # Also save locally for future runs
                            z.extract(name, self.output_dir)
                            break
                print("    ✅ Download complete!")
            except Exception as e:
                print(f"    ⚠  Download failed ({e}). Creating dataset from built-in samples...")
                self._create_fallback_dataset()

        # Encode labels: ham=0, spam=1
        self.df['label_encoded'] = self.label_encoder.fit_transform(self.df['label'])

        # Display summary
        section("Dataset Overview")
        kv("Total messages", f"{len(self.df):,}")
        kv("Features", "message (text)")
        kv("Target", "label (ham / spam)")
        kv("Columns", f"{list(self.df.columns)}")

        print(f"\n    📋 First 5 messages:")
        print(f"    {'─' * 56}")
        for i, row in self.df.head(5).iterrows():
            msg = row['message'][:55] + "..." if len(row['message']) > 55 else row['message']
            tag = "🟢 HAM " if row['label'] == 'ham' else "🔴 SPAM"
            print(f"    {tag}  │ {msg}")

        return self

    def _create_fallback_dataset(self):
        """Create a representative fallback dataset if download fails."""
        spam_msgs = [
            "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!",
            "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121",
            "Urgent! You have won a 1 week free membership in our £100,000 Prize Jackpot!",
            "CONGRATULATIONS! You've won $5000! Call now to claim your prize!",
            "FREE MSG: Claim your free iPhone now! Limited time offer, text WIN to 12345",
            "You are a winner U have been selected for a cash prize! Call 09061701461",
            "Congratulations ur awarded 500 of CD vouchers. Text COLLECT to 83600",
            "Had your mobile 11 months or more? U R entitled to update and claim free gift",
            "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575",
            "URGENT! Your Mobile No. was awarded a £2,000 Bonus Caller Prize",
        ] * 55  # ~550 spam messages

        ham_msgs = [
            "Hey, are you coming to the party tonight?",
            "I'll be there in 20 minutes. Wait for me.",
            "Can you pick up some milk on the way home?",
            "Great work on the presentation today!",
            "Let me know when you're free for lunch this week.",
            "Happy birthday! Hope you have a wonderful day!",
            "Thanks for helping me with the assignment yesterday.",
            "The meeting has been rescheduled to 3pm tomorrow.",
            "I just finished reading that book you recommended. It was amazing!",
            "Do you want to go hiking this weekend?",
            "Mom said dinner will be ready by 7pm.",
            "Just got home. How was your day?",
            "The weather looks great today, let's go for a walk!",
            "Can you send me the notes from today's lecture?",
            "I'll call you later tonight to discuss the plans.",
            "Sorry I missed your call. What's up?",
            "Remember to bring your laptop tomorrow for the project.",
            "That movie was so good! We should watch the sequel.",
            "I'm running late, save me a seat please.",
            "Good morning! Have a productive day ahead.",
        ] * 240  # ~4800 ham messages

        labels = ['spam'] * len(spam_msgs) + ['ham'] * len(ham_msgs)
        messages = spam_msgs + ham_msgs

        self.df = pd.DataFrame({'label': labels, 'message': messages})
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ───────────────────────────────────────────────────────────────
    # STEP 2: Exploratory Data Analysis
    # ───────────────────────────────────────────────────────────────

    def explore_data(self):
        """Perform exploratory data analysis with visualizations."""
        banner("STEP 2 ─ EXPLORATORY DATA ANALYSIS")

        # Class distribution
        section("Class Distribution")
        class_counts = self.df['label'].value_counts()
        total = len(self.df)
        for label, count in class_counts.items():
            pct = count / total * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            icon = "🟢" if label == "ham" else "🔴"
            kv(f"{icon} {label.upper()}", f"{count:>5} messages  ({pct:.1f}%)  {bar}")

        # Add text features for analysis
        self.df['msg_length'] = self.df['message'].apply(len)
        self.df['word_count'] = self.df['message'].apply(lambda x: len(x.split()))
        self.df['capital_count'] = self.df['message'].apply(lambda x: sum(1 for c in x if c.isupper()))
        self.df['digit_count'] = self.df['message'].apply(lambda x: sum(1 for c in x if c.isdigit()))
        self.df['special_count'] = self.df['message'].apply(
            lambda x: sum(1 for c in x if c in '!?$£€¥₹@#%&*')
        )
        self.df['capital_ratio'] = self.df['capital_count'] / (self.df['msg_length'] + 1)
        self.df['avg_word_length'] = self.df['message'].apply(
            lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
        )

        # Statistical comparison
        section("Ham vs Spam — Statistical Comparison")
        print(f"\n    {'Metric':<22} {'Ham':>12} {'Spam':>12} {'Ratio':>10}")
        print(f"    {'─' * 56}")

        for col_name, label in [
            ('msg_length', 'Avg Length (chars)'),
            ('word_count', 'Avg Word Count'),
            ('capital_count', 'Avg Capitals'),
            ('digit_count', 'Avg Digits'),
            ('special_count', 'Avg Special Chars'),
            ('avg_word_length', 'Avg Word Length'),
        ]:
            ham_val = self.df[self.df['label'] == 'ham'][col_name].mean()
            spam_val = self.df[self.df['label'] == 'spam'][col_name].mean()
            ratio = spam_val / ham_val if ham_val > 0 else 0
            print(f"    {label:<22} {ham_val:>12.1f} {spam_val:>12.1f} {ratio:>9.2f}x")

        # ── Visualization 1: Class Distribution ──
        self._plot_class_distribution(class_counts)

        # ── Visualization 2: Message Length Distribution ──
        self._plot_message_length_distribution()

        # ── Visualization 3: Word Count Distribution ──
        self._plot_word_count_distribution()

        # ── Visualization 4: Text Feature Comparison ──
        self._plot_feature_comparison()

        # ── Visualization 5: Word Clouds ──
        self._plot_word_clouds()

        # ── Visualization 6: Top Words ──
        self._plot_top_words()

        print("\n    ✅ 6 EDA visualizations saved!")
        return self

    def _plot_class_distribution(self, class_counts):
        """Plot 1: Stunning class distribution donut chart."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Donut chart
        colors = [PALETTE['ham_color'], PALETTE['spam_color']]
        wedges, texts, autotexts = axes[0].pie(
            class_counts.values, labels=['Ham', 'Spam'],
            autopct='%1.1f%%', colors=colors, startangle=90,
            pctdistance=0.75, explode=(0, 0.06),
            wedgeprops=dict(width=0.4, edgecolor=PALETTE['bg_dark'], linewidth=3),
            textprops=dict(color=PALETTE['text_light'], fontsize=14, fontweight='bold')
        )
        for autotext in autotexts:
            autotext.set_fontsize(13)
            autotext.set_fontweight('bold')
        axes[0].set_title('Class Distribution', fontsize=16, fontweight='bold',
                          color=PALETTE['text_light'], pad=15)
        # Center text
        axes[0].text(0, 0, f'{len(self.df):,}\nTotal', ha='center', va='center',
                     fontsize=16, fontweight='bold', color=PALETTE['accent_cyan'])

        # Bar chart
        bars = axes[1].bar(['Ham\n(Legitimate)', 'Spam\n(Unwanted)'],
                           class_counts.values, color=colors,
                           edgecolor=PALETTE['bg_dark'], linewidth=2,
                           width=0.55, alpha=0.9)
        for bar, count in zip(bars, class_counts.values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                         f'{count:,}', ha='center', va='bottom',
                         fontsize=14, fontweight='bold', color=PALETTE['accent_gold'])
        axes[1].set_ylabel('Number of Messages', fontsize=13)
        axes[1].set_title('Message Counts', fontsize=16, fontweight='bold',
                          color=PALETTE['text_light'], pad=15)
        axes[1].grid(True, axis='y', alpha=0.2)
        axes[1].set_axisbelow(True)

        plt.tight_layout(pad=2)
        plt.savefig(os.path.join(self.output_dir, '01_class_distribution.png'),
                    bbox_inches='tight')
        plt.close()

    def _plot_message_length_distribution(self):
        """Plot 2: Message length distributions by class."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ham_lens = self.df[self.df['label'] == 'ham']['msg_length']
        spam_lens = self.df[self.df['label'] == 'spam']['msg_length']

        # Histogram
        axes[0].hist(ham_lens, bins=50, alpha=0.7, color=PALETTE['ham_color'],
                     label='Ham', edgecolor='none', density=True)
        axes[0].hist(spam_lens, bins=50, alpha=0.7, color=PALETTE['spam_color'],
                     label='Spam', edgecolor='none', density=True)
        axes[0].set_xlabel('Message Length (characters)', fontsize=12)
        axes[0].set_ylabel('Density', fontsize=12)
        axes[0].set_title('Message Length Distribution', fontsize=15, fontweight='bold',
                          color=PALETTE['text_light'])
        axes[0].legend(frameon=True, fancybox=True)
        axes[0].grid(True, alpha=0.2)
        axes[0].set_axisbelow(True)

        # Box plot
        box_data = [ham_lens, spam_lens]
        bp = axes[1].boxplot(box_data, labels=['Ham', 'Spam'], patch_artist=True,
                             widths=0.45,
                             medianprops=dict(color=PALETTE['accent_gold'], linewidth=2.5),
                             whiskerprops=dict(color=PALETTE['text_muted'], linewidth=1.5),
                             capprops=dict(color=PALETTE['text_muted'], linewidth=1.5),
                             flierprops=dict(marker='o', markerfacecolor=PALETTE['accent_purple'],
                                             markersize=4, alpha=0.5))
        bp['boxes'][0].set_facecolor(PALETTE['ham_color'])
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor(PALETTE['spam_color'])
        bp['boxes'][1].set_alpha(0.6)
        axes[1].set_ylabel('Message Length', fontsize=12)
        axes[1].set_title('Length Comparison (Box Plot)', fontsize=15, fontweight='bold',
                          color=PALETTE['text_light'])
        axes[1].grid(True, axis='y', alpha=0.2)
        axes[1].set_axisbelow(True)

        plt.tight_layout(pad=2)
        plt.savefig(os.path.join(self.output_dir, '02_message_length.png'),
                    bbox_inches='tight')
        plt.close()

    def _plot_word_count_distribution(self):
        """Plot 3: Word count distribution with KDE overlay."""
        fig, ax = plt.subplots(figsize=(12, 6))

        ham_wc = self.df[self.df['label'] == 'ham']['word_count']
        spam_wc = self.df[self.df['label'] == 'spam']['word_count']

        # KDE plot
        ham_wc.plot.kde(ax=ax, color=PALETTE['ham_color'], linewidth=3,
                        label='Ham', alpha=0.9)
        spam_wc.plot.kde(ax=ax, color=PALETTE['spam_color'], linewidth=3,
                         label='Spam', alpha=0.9)

        # Fill under curves
        from scipy.stats import gaussian_kde
        x_range = np.linspace(0, max(ham_wc.max(), spam_wc.max()), 300)
        ham_kde = gaussian_kde(ham_wc)
        spam_kde = gaussian_kde(spam_wc)
        ax.fill_between(x_range, ham_kde(x_range), alpha=0.15, color=PALETTE['ham_color'])
        ax.fill_between(x_range, spam_kde(x_range), alpha=0.15, color=PALETTE['spam_color'])

        # Add mean lines
        ax.axvline(ham_wc.mean(), color=PALETTE['ham_color'], linestyle='--',
                   linewidth=2, alpha=0.8, label=f'Ham Mean: {ham_wc.mean():.0f}')
        ax.axvline(spam_wc.mean(), color=PALETTE['spam_color'], linestyle='--',
                   linewidth=2, alpha=0.8, label=f'Spam Mean: {spam_wc.mean():.0f}')

        ax.set_xlabel('Word Count per Message', fontsize=13)
        ax.set_ylabel('Density', fontsize=13)
        ax.set_title('Word Count Distribution — Ham vs Spam', fontsize=16, fontweight='bold',
                     color=PALETTE['text_light'])
        ax.legend(fontsize=11, frameon=True, fancybox=True)
        ax.grid(True, alpha=0.2)
        ax.set_axisbelow(True)
        ax.set_xlim(0, None)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '03_word_count_kde.png'),
                    bbox_inches='tight')
        plt.close()

    def _plot_feature_comparison(self):
        """Plot 4: Multi-feature radar / grouped bar comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        features = ['msg_length', 'word_count', 'capital_count', 'digit_count',
                     'special_count', 'avg_word_length']
        feature_names = ['Msg Length', 'Word Count', 'Capitals', 'Digits',
                         'Special Chars', 'Avg Word Len']

        ham_means = [self.df[self.df['label'] == 'ham'][f].mean() for f in features]
        spam_means = [self.df[self.df['label'] == 'spam'][f].mean() for f in features]

        # Normalize for radar plot
        max_vals = [max(h, s) for h, s in zip(ham_means, spam_means)]
        ham_norm = [h / m if m > 0 else 0 for h, m in zip(ham_means, max_vals)]
        spam_norm = [s / m if m > 0 else 0 for s, m in zip(spam_means, max_vals)]

        # Radar chart
        N = len(features)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        ham_norm += ham_norm[:1]
        spam_norm += spam_norm[:1]

        ax = axes[0]
        ax.set_facecolor(PALETTE['bg_card'])
        ax = fig.add_subplot(121, polar=True)
        ax.set_facecolor(PALETTE['bg_card'])

        ax.plot(angles, ham_norm, 'o-', linewidth=2.5, color=PALETTE['ham_color'],
                label='Ham', markersize=7)
        ax.fill(angles, ham_norm, alpha=0.15, color=PALETTE['ham_color'])
        ax.plot(angles, spam_norm, 'o-', linewidth=2.5, color=PALETTE['spam_color'],
                label='Spam', markersize=7)
        ax.fill(angles, spam_norm, alpha=0.15, color=PALETTE['spam_color'])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names, fontsize=9, color=PALETTE['text_light'])
        ax.set_title('Feature Profile Comparison', fontsize=14, fontweight='bold',
                     color=PALETTE['text_light'], pad=25)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10,
                  frameon=True, fancybox=True)
        ax.set_yticklabels([])
        ax.grid(True, color=PALETTE['grid'], alpha=0.4)

        # Grouped bar chart (absolute values)
        x = np.arange(len(feature_names))
        width = 0.35
        axes[1].bar(x - width / 2, ham_means, width, label='Ham',
                    color=PALETTE['ham_color'], alpha=0.85, edgecolor='none')
        axes[1].bar(x + width / 2, spam_means, width, label='Spam',
                    color=PALETTE['spam_color'], alpha=0.85, edgecolor='none')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(feature_names, rotation=30, ha='right', fontsize=10)
        axes[1].set_ylabel('Mean Value', fontsize=12)
        axes[1].set_title('Feature Means — Ham vs Spam', fontsize=14, fontweight='bold',
                          color=PALETTE['text_light'])
        axes[1].legend(fontsize=11, frameon=True, fancybox=True)
        axes[1].grid(True, axis='y', alpha=0.2)
        axes[1].set_axisbelow(True)

        plt.tight_layout(pad=2)
        plt.savefig(os.path.join(self.output_dir, '04_feature_comparison.png'),
                    bbox_inches='tight')
        plt.close()

    def _plot_word_clouds(self):
        """Plot 5: Word clouds for ham and spam messages."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        ham_text = ' '.join(self.df[self.df['label'] == 'ham']['message'].values)
        spam_text = ' '.join(self.df[self.df['label'] == 'spam']['message'].values)

        # Ham word cloud
        wc_ham = WordCloud(
            width=800, height=400, background_color=PALETTE['bg_dark'],
            colormap='Greens', max_words=120, max_font_size=80,
            contour_width=2, contour_color=PALETTE['ham_color'],
            random_state=42
        ).generate(ham_text)
        axes[0].imshow(wc_ham, interpolation='bilinear')
        axes[0].set_title('🟢 Ham Messages — Word Cloud', fontsize=15, fontweight='bold',
                          color=PALETTE['ham_color'], pad=12)
        axes[0].axis('off')

        # Spam word cloud
        wc_spam = WordCloud(
            width=800, height=400, background_color=PALETTE['bg_dark'],
            colormap='Reds', max_words=120, max_font_size=80,
            contour_width=2, contour_color=PALETTE['spam_color'],
            random_state=42
        ).generate(spam_text)
        axes[1].imshow(wc_spam, interpolation='bilinear')
        axes[1].set_title('🔴 Spam Messages — Word Cloud', fontsize=15, fontweight='bold',
                          color=PALETTE['spam_color'], pad=12)
        axes[1].axis('off')

        plt.tight_layout(pad=2)
        plt.savefig(os.path.join(self.output_dir, '05_word_clouds.png'),
                    bbox_inches='tight')
        plt.close()

    def _plot_top_words(self):
        """Plot 6: Top 15 most frequent words in ham vs spam."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        for idx, (label, color, title) in enumerate([
            ('ham', PALETTE['ham_color'], '🟢 Top 15 Words in Ham'),
            ('spam', PALETTE['spam_color'], '🔴 Top 15 Words in Spam')
        ]):
            subset = self.df[self.df['label'] == label]['message']
            all_words = ' '.join(subset).lower().split()
            # Remove common stopwords and punctuation
            filtered = [w.strip(string.punctuation) for w in all_words
                        if w.lower().strip(string.punctuation) not in self.stop_words
                        and len(w.strip(string.punctuation)) > 1]
            word_freq = Counter(filtered).most_common(15)
            words, counts = zip(*word_freq) if word_freq else ([], [])

            y_pos = np.arange(len(words))
            bars = axes[idx].barh(y_pos, counts, color=color, alpha=0.85,
                                  edgecolor='none', height=0.65)
            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels(words, fontsize=11)
            axes[idx].invert_yaxis()
            axes[idx].set_xlabel('Frequency', fontsize=12)
            axes[idx].set_title(title, fontsize=14, fontweight='bold',
                                color=PALETTE['text_light'])
            axes[idx].grid(True, axis='x', alpha=0.2)
            axes[idx].set_axisbelow(True)

            # Add count labels
            for bar, count in zip(bars, counts):
                axes[idx].text(bar.get_width() + max(counts) * 0.02,
                               bar.get_y() + bar.get_height() / 2,
                               f'{count:,}', va='center', fontsize=10,
                               color=PALETTE['text_muted'], fontweight='bold')

        plt.tight_layout(pad=2)
        plt.savefig(os.path.join(self.output_dir, '06_top_words.png'),
                    bbox_inches='tight')
        plt.close()

    # ───────────────────────────────────────────────────────────────
    # STEP 3: Text Preprocessing
    # ───────────────────────────────────────────────────────────────

    def preprocess_text(self):
        """Clean and preprocess all text messages."""
        banner("STEP 3 ─ TEXT PREPROCESSING")

        section("Preprocessing Pipeline")
        print("    1. Convert to lowercase")
        print("    2. Remove URLs, emails, phone numbers")
        print("    3. Remove HTML tags")
        print("    4. Remove punctuation & special characters")
        print("    5. Remove numbers")
        print("    6. Tokenize text")
        print("    7. Remove stopwords")
        print("    8. Apply stemming (Porter Stemmer)")

        total = len(self.df)
        self.df['cleaned'] = ''

        for i, row in self.df.iterrows():
            self.df.at[i, 'cleaned'] = self._clean_text(row['message'])
            if (i + 1) % 500 == 0 or i == total - 1:
                progress(i + 1, total, f"Processing message {i+1}/{total}")

        # Show examples
        section("Before vs After Preprocessing")
        sample = self.df.sample(3, random_state=42)
        for _, row in sample.iterrows():
            tag = "🟢 HAM " if row['label'] == 'ham' else "🔴 SPAM"
            print(f"\n    {tag}")
            orig = row['message'][:70] + "..." if len(row['message']) > 70 else row['message']
            clean = row['cleaned'][:70] + "..." if len(row['cleaned']) > 70 else row['cleaned']
            print(f"    BEFORE │ {orig}")
            print(f"    AFTER  │ {clean}")

        # Add cleaned text stats
        self.df['cleaned_length'] = self.df['cleaned'].apply(len)
        self.df['cleaned_word_count'] = self.df['cleaned'].apply(lambda x: len(x.split()))

        section("Preprocessing Impact")
        kv("Avg original length", f"{self.df['msg_length'].mean():.1f} chars")
        kv("Avg cleaned length", f"{self.df['cleaned_length'].mean():.1f} chars")
        kv("Reduction", f"{(1 - self.df['cleaned_length'].mean() / self.df['msg_length'].mean()) * 100:.1f}%")
        kv("Empty after cleaning", f"{(self.df['cleaned_length'] == 0).sum()}")

        print("\n    ✅ Text preprocessing complete!")
        return self

    def _clean_text(self, text):
        """Apply full cleaning pipeline to a single text message."""
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove phone numbers
        text = re.sub(r'\b\d{10,}\b', '', text)
        text = re.sub(r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]*', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove punctuation and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = text.split()
        # Remove stopwords and stem
        tokens = [
            self.stemmer.stem(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 1
        ]
        return ' '.join(tokens)

    # ───────────────────────────────────────────────────────────────
    # STEP 4: Feature Extraction (TF-IDF)
    # ───────────────────────────────────────────────────────────────

    def extract_features(self):
        """Convert cleaned text into TF-IDF feature vectors."""
        banner("STEP 4 ─ FEATURE EXTRACTION (TF-IDF)")

        # Split data first
        section("Train-Test Split")
        X = self.df['cleaned']
        y = self.df['label_encoded']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        kv("Training set", f"{len(self.X_train):,} messages ({len(self.X_train)/len(X)*100:.0f}%)")
        kv("Test set", f"{len(self.X_test):,} messages ({len(self.X_test)/len(X)*100:.0f}%)")
        kv("Train — Ham", f"{(self.y_train == 0).sum():,}")
        kv("Train — Spam", f"{(self.y_train == 1).sum():,}")
        kv("Test  — Ham", f"{(self.y_test == 0).sum():,}")
        kv("Test  — Spam", f"{(self.y_test == 1).sum():,}")

        # TF-IDF Vectorization
        section("TF-IDF Vectorization")
        self.tfidf = TfidfVectorizer(
            max_features=5000,       # Top 5000 features
            ngram_range=(1, 2),      # Unigrams + bigrams
            min_df=2,                # Minimum document frequency
            max_df=0.95,             # Maximum document frequency
            sublinear_tf=True,       # Apply sublinear TF scaling
            strip_accents='unicode',
            dtype=np.float32
        )

        self.X_train_tfidf = self.tfidf.fit_transform(self.X_train)
        self.X_test_tfidf = self.tfidf.transform(self.X_test)

        kv("Vocabulary size", f"{len(self.tfidf.vocabulary_):,} terms")
        kv("Feature matrix (train)", f"{self.X_train_tfidf.shape}")
        kv("Feature matrix (test)", f"{self.X_test_tfidf.shape}")
        kv("Sparsity", f"{(1 - self.X_train_tfidf.nnz / (self.X_train_tfidf.shape[0] * self.X_train_tfidf.shape[1])) * 100:.2f}%")
        kv("N-gram range", "Unigrams + Bigrams (1, 2)")
        kv("Sublinear TF", "Enabled (log normalization)")

        # Show top TF-IDF features
        feature_names = self.tfidf.get_feature_names_out()
        avg_tfidf = np.array(self.X_train_tfidf.mean(axis=0)).flatten()
        top_indices = avg_tfidf.argsort()[-10:][::-1]

        section("Top 10 TF-IDF Features (by average score)")
        for rank, idx in enumerate(top_indices, 1):
            print(f"    {rank:>2}. {feature_names[idx]:<25}  TF-IDF: {avg_tfidf[idx]:.4f}")

        print("\n    ✅ Feature extraction complete!")
        return self

    # ───────────────────────────────────────────────────────────────
    # STEP 5: Model Training & Comparison
    # ───────────────────────────────────────────────────────────────

    def train_models(self):
        """Train multiple classifiers and compare performance."""
        banner("STEP 5 ─ MODEL TRAINING & COMPARISON")

        # Define models
        self.models = {
            'Multinomial Naive Bayes': MultinomialNB(alpha=1.0),
            'Complement Naive Bayes': ComplementNB(alpha=1.0),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=42, C=1.0, solver='lbfgs'
            ),
            'Linear SVM': LinearSVC(
                max_iter=2000, random_state=42, C=1.0
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150, random_state=42, max_depth=5
            ),
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        section("Training 6 Models with 5-Fold Cross-Validation")
        print(f"\n    {'Model':<28} {'CV Acc':>8} {'CV Std':>8} {'Test Acc':>10} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        print(f"    {'─' * 88}")

        for name, model in self.models.items():
            # Cross-validation
            cv_scores = cross_val_score(
                model, self.X_train_tfidf, self.y_train,
                cv=cv, scoring='accuracy', n_jobs=-1
            )

            # Train on full training set
            model.fit(self.X_train_tfidf, self.y_train)
            y_pred = model.predict(self.X_test_tfidf)

            # Metrics
            test_acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred)
            rec = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)

            self.results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores,
                'test_accuracy': test_acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'y_pred': y_pred
            }

            print(f"    {name:<28} {cv_scores.mean():>7.4f}  ±{cv_scores.std():>6.4f}  "
                  f"{test_acc:>9.4f}  {prec:>7.4f}  {rec:>7.4f}  {f1:>7.4f}")

        # Find best model by F1 score (better for imbalanced datasets)
        self.best_model_name = max(self.results, key=lambda k: self.results[k]['f1'])
        self.best_model = self.results[self.best_model_name]['model']

        print(f"\n    🏆 Best Model (by F1 Score): {self.best_model_name}")
        print(f"       F1: {self.results[self.best_model_name]['f1']:.4f}  |  "
              f"Accuracy: {self.results[self.best_model_name]['test_accuracy']:.4f}")

        # ── Visualization 7: Model Comparison ──
        self._plot_model_comparison()

        print("\n    ✅ All models trained and compared!")
        return self

    def _plot_model_comparison(self):
        """Plot 7: Multi-metric model comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        names = list(self.results.keys())
        short = ['MNB', 'CNB', 'LogReg', 'SVM', 'RF', 'GBM']
        metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_keys = ['test_accuracy', 'precision', 'recall', 'f1']
        colors = [PALETTE['accent_cyan'], PALETTE['accent_gold'],
                  PALETTE['accent_green'], PALETTE['accent_magenta']]

        # Grouped bar chart
        x = np.arange(len(names))
        width = 0.18
        for i, (metric_key, metric_label, color) in enumerate(
                zip(metric_keys, metrics_labels, colors)):
            values = [self.results[n][metric_key] for n in names]
            axes[0].bar(x + i * width - 1.5 * width, values, width,
                        label=metric_label, color=color, alpha=0.85, edgecolor='none')

        axes[0].set_xticks(x)
        axes[0].set_xticklabels(short, fontsize=10)
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].set_title('Model Performance — All Metrics', fontsize=15,
                          fontweight='bold', color=PALETTE['text_light'])
        axes[0].legend(fontsize=9, frameon=True, fancybox=True, ncol=2)
        axes[0].set_ylim(0.7, 1.05)
        axes[0].grid(True, axis='y', alpha=0.2)
        axes[0].set_axisbelow(True)

        # CV Box plots
        cv_data = [self.results[n]['cv_scores'] for n in names]
        bp = axes[1].boxplot(cv_data, labels=short, patch_artist=True,
                             widths=0.45,
                             medianprops=dict(color=PALETTE['accent_gold'], linewidth=2.5),
                             whiskerprops=dict(color=PALETTE['text_muted'], linewidth=1.5),
                             capprops=dict(color=PALETTE['text_muted'], linewidth=1.5))
        box_colors = [PALETTE['accent_cyan'], PALETTE['accent_green'],
                      PALETTE['accent_gold'], PALETTE['accent_magenta'],
                      PALETTE['accent_purple'], PALETTE['ham_color']]
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        axes[1].set_ylabel('CV Accuracy', fontsize=12)
        axes[1].set_title('Cross-Validation Score Distribution', fontsize=15,
                          fontweight='bold', color=PALETTE['text_light'])
        axes[1].set_ylim(0.85, 1.02)
        axes[1].grid(True, axis='y', alpha=0.2)
        axes[1].set_axisbelow(True)

        plt.tight_layout(pad=2)
        plt.savefig(os.path.join(self.output_dir, '07_model_comparison.png'),
                    bbox_inches='tight')
        plt.close()

    # ───────────────────────────────────────────────────────────────
    # STEP 6: Detailed Evaluation
    # ───────────────────────────────────────────────────────────────

    def evaluate(self):
        """Comprehensive evaluation of the best model."""
        banner("STEP 6 ─ DETAILED EVALUATION")

        y_pred = self.results[self.best_model_name]['y_pred']

        section(f"Best Model: {self.best_model_name}")

        # Classification Report
        print("\n    📋 Classification Report:")
        print(f"    {'─' * 55}")
        report = classification_report(
            self.y_test, y_pred,
            target_names=['Ham', 'Spam'], digits=4
        )
        for line in report.split('\n'):
            print(f"    {line}")

        # Summary metrics
        section("Summary Metrics")
        acc = self.results[self.best_model_name]['test_accuracy']
        prec = self.results[self.best_model_name]['precision']
        rec = self.results[self.best_model_name]['recall']
        f1 = self.results[self.best_model_name]['f1']

        kv("Accuracy", f"{acc:.4f}  ({acc*100:.2f}%)")
        kv("Precision (Spam)", f"{prec:.4f}")
        kv("Recall (Spam)", f"{rec:.4f}")
        kv("F1-Score (Spam)", f"{f1:.4f}")

        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        kv("True Positives (Spam→Spam)", f"{tp}")
        kv("True Negatives (Ham→Ham)", f"{tn}")
        kv("False Positives (Ham→Spam)", f"{fp}")
        kv("False Negatives (Spam→Ham)", f"{fn}")

        # ── Visualization 8: Confusion Matrix ──
        self._plot_confusion_matrix(cm, y_pred)

        # ── Visualization 9: ROC Curve ──
        self._plot_roc_curves()

        # ── Visualization 10: Precision-Recall Curve ──
        self._plot_precision_recall_curves()

        # ── Visualization 11: Learning Curves ──
        self._plot_learning_curves()

        # ── Visualization 12: Feature Importance ──
        self._plot_top_features()

        print("\n    ✅ Evaluation complete — 5 evaluation visualizations saved!")
        return self

    def _plot_confusion_matrix(self, cm, y_pred):
        """Plot 8: Dual confusion matrix (counts + percentages)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        labels = ['Ham', 'Spam']

        # Counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    ax=axes[0], linewidths=2, linecolor=PALETTE['bg_dark'],
                    annot_kws={'size': 20, 'fontweight': 'bold'},
                    cbar_kws={'shrink': 0.8})
        axes[0].set_xlabel('Predicted Label', fontsize=13, color=PALETTE['text_light'])
        axes[0].set_ylabel('True Label', fontsize=13, color=PALETTE['text_light'])
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=15, fontweight='bold',
                          color=PALETTE['text_light'])

        # Normalized percentages
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Purples',
                    xticklabels=labels, yticklabels=labels,
                    ax=axes[1], linewidths=2, linecolor=PALETTE['bg_dark'],
                    annot_kws={'size': 18, 'fontweight': 'bold'},
                    cbar_kws={'shrink': 0.8, 'format': '%.0f%%'})
        axes[1].set_xlabel('Predicted Label', fontsize=13, color=PALETTE['text_light'])
        axes[1].set_ylabel('True Label', fontsize=13, color=PALETTE['text_light'])
        axes[1].set_title('Confusion Matrix (Normalized %)', fontsize=15, fontweight='bold',
                          color=PALETTE['text_light'])

        plt.tight_layout(pad=2)
        plt.savefig(os.path.join(self.output_dir, '08_confusion_matrix.png'),
                    bbox_inches='tight')
        plt.close()

    def _plot_roc_curves(self):
        """Plot 9: ROC curves for all models that support predict_proba or decision_function."""
        fig, ax = plt.subplots(figsize=(10, 8))

        colors_list = [PALETTE['accent_cyan'], PALETTE['accent_green'],
                       PALETTE['accent_gold'], PALETTE['accent_magenta'],
                       PALETTE['accent_purple'], PALETTE['ham_color']]

        for idx, (name, res) in enumerate(self.results.items()):
            model = res['model']
            try:
                if hasattr(model, 'predict_proba'):
                    y_scores = model.predict_proba(self.X_test_tfidf)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(self.X_test_tfidf)
                else:
                    continue

                fpr, tpr, _ = roc_curve(self.y_test, y_scores)
                roc_auc = auc(fpr, tpr)

                lw = 3.5 if name == self.best_model_name else 2
                alpha = 1.0 if name == self.best_model_name else 0.7
                ls = '-' if name == self.best_model_name else '--'

                ax.plot(fpr, tpr, color=colors_list[idx % len(colors_list)],
                        linewidth=lw, alpha=alpha, linestyle=ls,
                        label=f'{name} (AUC = {roc_auc:.4f})')
            except Exception:
                continue

        # Diagonal
        ax.plot([0, 1], [0, 1], 'w--', linewidth=1.5, alpha=0.4, label='Random (AUC = 0.5)')

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('False Positive Rate', fontsize=13)
        ax.set_ylabel('True Positive Rate', fontsize=13)
        ax.set_title('ROC Curves — All Models', fontsize=16, fontweight='bold',
                     color=PALETTE['text_light'])
        ax.legend(loc='lower right', fontsize=9, frameon=True, fancybox=True)
        ax.grid(True, alpha=0.2)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '09_roc_curves.png'),
                    bbox_inches='tight')
        plt.close()

    def _plot_precision_recall_curves(self):
        """Plot 10: Precision-Recall curves."""
        fig, ax = plt.subplots(figsize=(10, 8))

        colors_list = [PALETTE['accent_cyan'], PALETTE['accent_green'],
                       PALETTE['accent_gold'], PALETTE['accent_magenta'],
                       PALETTE['accent_purple'], PALETTE['ham_color']]

        for idx, (name, res) in enumerate(self.results.items()):
            model = res['model']
            try:
                if hasattr(model, 'predict_proba'):
                    y_scores = model.predict_proba(self.X_test_tfidf)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(self.X_test_tfidf)
                else:
                    continue

                precision_vals, recall_vals, _ = precision_recall_curve(
                    self.y_test, y_scores)
                ap = average_precision_score(self.y_test, y_scores)

                lw = 3.5 if name == self.best_model_name else 2
                alpha = 1.0 if name == self.best_model_name else 0.7

                ax.plot(recall_vals, precision_vals,
                        color=colors_list[idx % len(colors_list)],
                        linewidth=lw, alpha=alpha,
                        label=f'{name} (AP = {ap:.4f})')
            except Exception:
                continue

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel('Recall', fontsize=13)
        ax.set_ylabel('Precision', fontsize=13)
        ax.set_title('Precision-Recall Curves — All Models', fontsize=16,
                     fontweight='bold', color=PALETTE['text_light'])
        ax.legend(loc='lower left', fontsize=9, frameon=True, fancybox=True)
        ax.grid(True, alpha=0.2)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '10_precision_recall.png'),
                    bbox_inches='tight')
        plt.close()

    def _plot_learning_curves(self):
        """Plot 11: Learning curves for top 3 models."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        top_3 = sorted(self.results.items(), key=lambda x: x[1]['f1'], reverse=True)[:3]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        colors_pair = [
            (PALETTE['accent_cyan'], PALETTE['accent_magenta']),
            (PALETTE['accent_green'], PALETTE['accent_gold']),
            (PALETTE['accent_purple'], PALETTE['ham_color']),
        ]

        for idx, ((name, res), (c_train, c_val)) in enumerate(zip(top_3, colors_pair)):
            model = res['model'].__class__(**res['model'].get_params())

            train_sizes, train_scores, val_scores = learning_curve(
                model, self.X_train_tfidf, self.y_train,
                train_sizes=np.linspace(0.1, 1.0, 8),
                cv=cv, scoring='accuracy', n_jobs=-1
            )

            train_mean = train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            val_mean = val_scores.mean(axis=1)
            val_std = val_scores.std(axis=1)

            axes[idx].fill_between(train_sizes, train_mean - train_std,
                                    train_mean + train_std, alpha=0.15, color=c_train)
            axes[idx].fill_between(train_sizes, val_mean - val_std,
                                    val_mean + val_std, alpha=0.15, color=c_val)
            axes[idx].plot(train_sizes, train_mean, 'o-', color=c_train,
                           linewidth=2.5, markersize=6, label='Training')
            axes[idx].plot(train_sizes, val_mean, 's-', color=c_val,
                           linewidth=2.5, markersize=6, label='Validation')

            axes[idx].set_xlabel('Training Set Size', fontsize=12)
            axes[idx].set_ylabel('Accuracy', fontsize=12)
            short_name = name.split('(')[0].strip()
            axes[idx].set_title(short_name, fontsize=14, fontweight='bold',
                                color=PALETTE['text_light'])
            axes[idx].legend(loc='lower right', fontsize=10, frameon=True, fancybox=True)
            axes[idx].grid(True, alpha=0.2)
            axes[idx].set_axisbelow(True)
            axes[idx].set_ylim(0.85, 1.02)

        fig.suptitle('Learning Curves — Top 3 Models', fontsize=17, fontweight='bold',
                     color=PALETTE['accent_gold'], y=1.03)
        plt.tight_layout(pad=2)
        plt.savefig(os.path.join(self.output_dir, '11_learning_curves.png'),
                    bbox_inches='tight')
        plt.close()

    def _plot_top_features(self):
        """Plot 12: Most important features (words) for spam detection."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        feature_names = self.tfidf.get_feature_names_out()

        # Use Logistic Regression coefficients for interpretability
        lr_model = self.results.get('Logistic Regression', {}).get('model')
        if lr_model is not None and hasattr(lr_model, 'coef_'):
            coefs = lr_model.coef_[0]
        elif hasattr(self.best_model, 'coef_'):
            coefs = self.best_model.coef_[0] if len(self.best_model.coef_.shape) > 1 else self.best_model.coef_
        else:
            # Fallback: use Naive Bayes log probabilities
            nb_model = self.results.get('Multinomial Naive Bayes', {}).get('model')
            if nb_model and hasattr(nb_model, 'feature_log_prob_'):
                coefs = nb_model.feature_log_prob_[1] - nb_model.feature_log_prob_[0]
            else:
                print("    ⚠ Could not extract feature importances.")
                plt.close()
                return

        # Top spam indicators
        top_spam_idx = np.argsort(coefs)[-15:][::-1]
        top_spam_words = [feature_names[i] for i in top_spam_idx]
        top_spam_scores = [coefs[i] for i in top_spam_idx]

        bars1 = axes[0].barh(range(len(top_spam_words)), top_spam_scores,
                              color=PALETTE['spam_color'], alpha=0.85,
                              edgecolor='none', height=0.65)
        axes[0].set_yticks(range(len(top_spam_words)))
        axes[0].set_yticklabels(top_spam_words, fontsize=11)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Coefficient Weight', fontsize=12)
        axes[0].set_title('🔴 Top 15 Spam Indicators', fontsize=14, fontweight='bold',
                          color=PALETTE['spam_color'])
        axes[0].grid(True, axis='x', alpha=0.2)
        axes[0].set_axisbelow(True)

        # Top ham indicators
        top_ham_idx = np.argsort(coefs)[:15]
        top_ham_words = [feature_names[i] for i in top_ham_idx]
        top_ham_scores = [abs(coefs[i]) for i in top_ham_idx]

        bars2 = axes[1].barh(range(len(top_ham_words)), top_ham_scores,
                              color=PALETTE['ham_color'], alpha=0.85,
                              edgecolor='none', height=0.65)
        axes[1].set_yticks(range(len(top_ham_words)))
        axes[1].set_yticklabels(top_ham_words, fontsize=11)
        axes[1].invert_yaxis()
        axes[1].set_xlabel('|Coefficient Weight|', fontsize=12)
        axes[1].set_title('🟢 Top 15 Ham Indicators', fontsize=14, fontweight='bold',
                          color=PALETTE['ham_color'])
        axes[1].grid(True, axis='x', alpha=0.2)
        axes[1].set_axisbelow(True)

        fig.suptitle('Feature Importance — What Makes a Message Spam or Ham?',
                     fontsize=17, fontweight='bold', color=PALETTE['accent_gold'], y=1.02)
        plt.tight_layout(pad=2)
        plt.savefig(os.path.join(self.output_dir, '12_feature_importance.png'),
                    bbox_inches='tight')
        plt.close()

    # ───────────────────────────────────────────────────────────────
    # STEP 7: Final Summary
    # ───────────────────────────────────────────────────────────────

    def summary(self):
        """Print final summary of the entire pipeline."""
        banner("FINAL SUMMARY ─ SPAM MAIL DETECTOR")

        section("Pipeline Results")
        kv("Dataset", "SMS Spam Collection (UCI)")
        kv("Total messages", f"{len(self.df):,}")
        kv("Ham / Spam split", f"{(self.df['label']=='ham').sum():,} / {(self.df['label']=='spam').sum():,}")
        kv("Preprocessing", "Lowercase → Clean → Tokenize → Stem")
        kv("Feature extraction", f"TF-IDF ({self.X_train_tfidf.shape[1]:,} features)")
        kv("Models trained", f"{len(self.models)}")

        section("Best Model Performance")
        best = self.results[self.best_model_name]
        kv("Model", self.best_model_name)
        kv("Accuracy", f"{best['test_accuracy']:.4f}  ({best['test_accuracy']*100:.2f}%)")
        kv("Precision (Spam)", f"{best['precision']:.4f}")
        kv("Recall (Spam)", f"{best['recall']:.4f}")
        kv("F1-Score (Spam)", f"{best['f1']:.4f}")
        kv("CV Mean Accuracy", f"{best['cv_mean']:.4f} ± {best['cv_std']:.4f}")

        section("All Model Rankings (by F1 Score)")
        ranked = sorted(self.results.items(), key=lambda x: x[1]['f1'], reverse=True)
        print(f"\n    {'Rank':<6} {'Model':<28} {'F1':>8} {'Accuracy':>10} {'Precision':>10} {'Recall':>8}")
        print(f"    {'─' * 72}")
        for rank, (name, res) in enumerate(ranked, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"    {medal} {rank}   {name:<28} {res['f1']:>7.4f}  {res['test_accuracy']:>9.4f}"
                  f"  {res['precision']:>9.4f}  {res['recall']:>7.4f}")

        section("Visualizations Generated")
        viz_files = [f for f in os.listdir(self.output_dir) if f.endswith('.png')]
        viz_files.sort()
        for f in viz_files:
            print(f"    📊 {f}")

        section("Skills Demonstrated")
        skills = [
            "Text preprocessing (cleaning, tokenization, stemming)",
            "Feature extraction (TF-IDF vectorization with bigrams)",
            "Basic NLP (stopword removal, text normalization)",
            "Binary classification (6 ML algorithms compared)",
            "Model evaluation (accuracy, precision, recall, F1, ROC, PR curves)",
            "Data visualization (12 professional plots with dark theme)",
        ]
        for s in skills:
            print(f"    ✓ {s}")

        print()
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║               ✅  TASK 2 COMPLETE — SPAM MAIL DETECTOR              ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        print()

        return self


# ═══════════════════════════════════════════════════════════════════════
# MAIN — Run the complete pipeline
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                    📧  SPAM MAIL DETECTOR  📧                       ║")
    print("║              QSkill AI & ML Internship — Task 2                     ║")
    print("║                                                                      ║")
    print("║  Classifying SMS messages as Spam or Ham using NLP + ML             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    pipeline = SpamDetectorPipeline()

    # Execute the full pipeline
    (
        pipeline
        .load_data()
        .explore_data()
        .preprocess_text()
        .extract_features()
        .train_models()
        .evaluate()
        .summary()
    )
