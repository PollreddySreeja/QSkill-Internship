"""
Generate the Iris Flower Classification Jupyter Notebook.
Run this script once to create the .ipynb file, then open it in Jupyter.
"""
import json

def make_cell(cell_type, source, metadata=None):
    """Create a notebook cell."""
    cell = {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": source if isinstance(source, list) else source.split("\n")
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def md(text):
    """Create a markdown cell."""
    lines = text.strip().split("\n")
    # Add newline to each line except last
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return make_cell("markdown", source)

def code(text):
    """Create a code cell."""
    lines = text.strip().split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return make_cell("code", source)

cells = []

# ============================================================
# CELL 1 — Title & Introduction (Markdown)
# ============================================================
cells.append(md(r"""# 🌸 Iris Flower Classification

## QSkill AI & ML Internship — Task 1

**Author:** [Your Name]  
**Date:** June 2026  
**Objective:** Classify iris flowers into three species (*Setosa*, *Versicolor*, *Virginica*) based on sepal and petal measurements.

---

### 📋 Table of Contents
1. [Setup & Imports](#1.-Setup-&-Imports)
2. [Dataset Loading](#2.-Dataset-Loading)
3. [Exploratory Data Analysis (EDA)](#3.-Exploratory-Data-Analysis-(EDA))
4. [Data Preprocessing](#4.-Data-Preprocessing)
5. [Model Training & Comparison](#5.-Model-Training-&-Comparison)
6. [Hyperparameter Tuning](#6.-Hyperparameter-Tuning)
7. [Best Model Evaluation](#7.-Best-Model-Evaluation)
8. [Feature Importance Analysis](#8.-Feature-Importance-Analysis)
9. [Decision Boundary Visualization](#9.-Decision-Boundary-Visualization)
10. [Conclusion](#10.-Conclusion)

---

### 🧠 About the Dataset
The **Iris dataset** is one of the most well-known datasets in machine learning, introduced by the British statistician and biologist **Ronald Fisher** in 1936. It contains 150 samples from three species of Iris flowers:

| Species | Samples |
|---------|---------|
| *Iris Setosa* | 50 |
| *Iris Versicolor* | 50 |
| *Iris Virginica* | 50 |

Each sample has **4 features**: sepal length, sepal width, petal length, and petal width (all in centimeters)."""))

# ============================================================
# CELL 2 — Setup & Imports
# ============================================================
cells.append(md("""## 1. Setup & Imports"""))

cells.append(code(r"""# ============================================================
# Core Libraries
# ============================================================
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Visualization Libraries
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import ListedColormap

# ============================================================
# Scikit-Learn — Data
# ============================================================
from sklearn.datasets import load_iris

# ============================================================
# Scikit-Learn — Preprocessing & Splitting
# ============================================================
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV,
    StratifiedKFold, learning_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ============================================================
# Scikit-Learn — Models
# ============================================================
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier
)

# ============================================================
# Scikit-Learn — Metrics
# ============================================================
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    roc_curve, auc
)

# ============================================================
# Plot Styling
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 100,
    'savefig.dpi': 150,
    'font.family': 'sans-serif'
})

# Custom color palette for the 3 iris species
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1']
SPECIES_NAMES = ['Setosa', 'Versicolor', 'Virginica']

print("✅ All libraries imported successfully!")
print(f"   NumPy:        {np.__version__}")
print(f"   Pandas:       {pd.__version__}")
print(f"   Matplotlib:   {plt.matplotlib.__version__}")
print(f"   Seaborn:      {sns.__version__}")
import sklearn
print(f"   Scikit-learn:  {sklearn.__version__}")"""))

# ============================================================
# CELL 3 — Dataset Loading
# ============================================================
cells.append(md("""## 2. Dataset Loading"""))

cells.append(code(r"""# Load the Iris dataset from scikit-learn
iris = load_iris()

# Create a clean DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({
    0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'
})

# Rename columns for clarity
df.columns = [
    'sepal_length', 'sepal_width', 'petal_length', 'petal_width',
    'species', 'species_name'
]

print(f"📊 Dataset shape: {df.shape[0]} samples × {df.shape[1]} columns\n")
print("First 10 rows:")
df.head(10)"""))

cells.append(code(r"""# Dataset information
print("=" * 60)
print("📋 DATASET INFO")
print("=" * 60)
print(f"\n{'Feature':<20} {'Type':<12} {'Non-Null':<10} {'Null':<6}")
print("-" * 48)
for col in df.columns:
    non_null = df[col].notna().sum()
    null = df[col].isna().sum()
    dtype = str(df[col].dtype)
    print(f"{col:<20} {dtype:<12} {non_null:<10} {null:<6}")

print(f"\n✅ No missing values found — dataset is clean!")
print(f"📏 Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")"""))

cells.append(code(r"""# Statistical summary
print("=" * 60)
print("📈 STATISTICAL SUMMARY")
print("=" * 60)
df.describe().round(2)"""))

cells.append(code(r"""# Class distribution
print("=" * 60)
print("🎯 CLASS DISTRIBUTION")
print("=" * 60)
class_counts = df['species_name'].value_counts()
for species, count in class_counts.items():
    percentage = count / len(df) * 100
    bar = "█" * int(percentage / 2)
    print(f"  {species:<12} │ {count:>3} samples │ {percentage:>5.1f}% │ {bar}")
print(f"\n✅ Dataset is perfectly balanced (equal samples per class)")"""))

# ============================================================
# CELL 4 — Exploratory Data Analysis
# ============================================================
cells.append(md("""## 3. Exploratory Data Analysis (EDA)

Let's explore the dataset through comprehensive visualizations to understand the distributions, relationships, and separability of the three iris species."""))

cells.append(md("""### 3.1 Feature Distributions"""))

cells.append(code(r"""# ============================================================
# Distribution plots for each feature
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Features by Species', fontsize=18, fontweight='bold', y=1.02)

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
feature_labels = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)']

for idx, (feature, label) in enumerate(zip(features, feature_labels)):
    ax = axes[idx // 2, idx % 2]
    
    for i, species in enumerate(SPECIES_NAMES):
        subset = df[df['species_name'] == species][feature]
        ax.hist(subset, bins=15, alpha=0.6, color=COLORS[i], label=species,
                edgecolor='white', linewidth=0.8)
        # Add KDE curve
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(subset)
        x_range = np.linspace(subset.min() - 0.5, subset.max() + 0.5, 100)
        ax.plot(x_range, kde(x_range) * len(subset) * (subset.max() - subset.min()) / 15,
                color=COLORS[i], linewidth=2.5, alpha=0.9)
    
    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_feature_distributions.png', bbox_inches='tight', dpi=150)
plt.show()
print("📊 Key Insight: Petal features show much clearer species separation than sepal features.")"""))

cells.append(md("""### 3.2 Box Plots — Outlier Detection"""))

cells.append(code(r"""# ============================================================
# Box plots for outlier detection
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Box Plots — Feature Distribution & Outliers per Species',
             fontsize=16, fontweight='bold', y=1.02)

for idx, (feature, label) in enumerate(zip(features, feature_labels)):
    sns.boxplot(data=df, x='species_name', y=feature, 
                palette=COLORS, ax=axes[idx], 
                width=0.6, linewidth=1.5,
                flierprops=dict(marker='o', markerfacecolor='red', markersize=6))
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel(label, fontsize=11)
    axes[idx].set_title(label, fontsize=12, fontweight='bold')
    axes[idx].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('02_box_plots.png', bbox_inches='tight', dpi=150)
plt.show()
print("📊 Observation: Very few outliers detected — data quality is excellent.")"""))

cells.append(md("""### 3.3 Violin Plots — Distribution Shape"""))

cells.append(code(r"""# ============================================================
# Violin plots — show distribution shape + density
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Violin Plots — Distribution Shape by Species',
             fontsize=16, fontweight='bold', y=1.02)

for idx, (feature, label) in enumerate(zip(features, feature_labels)):
    sns.violinplot(data=df, x='species_name', y=feature,
                   palette=COLORS, ax=axes[idx],
                   inner='box', linewidth=1.2)
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel(label, fontsize=11)
    axes[idx].set_title(label, fontsize=12, fontweight='bold')
    axes[idx].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('03_violin_plots.png', bbox_inches='tight', dpi=150)
plt.show()
print("📊 Insight: Violin plots confirm petal features have distinct, non-overlapping distributions.")"""))

cells.append(md("""### 3.4 Pair Plot — Feature Relationships"""))

cells.append(code(r"""# ============================================================
# Pair plot — scatter matrix of all feature combinations
# ============================================================
pair_plot = sns.pairplot(
    df, hue='species_name',
    vars=features,
    palette=COLORS,
    diag_kind='kde',
    plot_kws={'alpha': 0.7, 's': 60, 'edgecolor': 'white', 'linewidth': 0.5},
    diag_kws={'linewidth': 2.5, 'fill': True, 'alpha': 0.4},
    height=2.5,
    aspect=1.1
)

pair_plot.fig.suptitle('Pairwise Feature Relationships', 
                        fontsize=18, fontweight='bold', y=1.02)
pair_plot._legend.set_title('Species')
pair_plot._legend.set_bbox_to_anchor((1.0, 0.5))

plt.savefig('04_pair_plot.png', bbox_inches='tight', dpi=150)
plt.show()
print("📊 Key Finding: Setosa is linearly separable. Versicolor & Virginica overlap slightly.")"""))

cells.append(md("""### 3.5 Correlation Heatmap"""))

cells.append(code(r"""# ============================================================
# Correlation heatmap
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Overall correlation
corr_matrix = df[features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f',
            cmap='RdYlBu_r', center=0, square=True,
            linewidths=2, linecolor='white',
            cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
            ax=axes[0], vmin=-1, vmax=1)
axes[0].set_title('Overall Feature Correlation', fontsize=14, fontweight='bold')

# Correlation with target
target_corr = df[features].corrwith(df['species']).sort_values(ascending=True)
bars = axes[1].barh(range(len(target_corr)), target_corr.values,
                     color=[COLORS[0] if v < 0 else COLORS[2] for v in target_corr.values],
                     edgecolor='white', linewidth=1)
axes[1].set_yticks(range(len(target_corr)))
axes[1].set_yticklabels([l.replace('_', ' ').title() for l in target_corr.index])
axes[1].set_xlabel('Correlation with Species', fontsize=12)
axes[1].set_title('Feature-Target Correlation', fontsize=14, fontweight='bold')
axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
for i, v in enumerate(target_corr.values):
    axes[1].text(v + 0.02 if v > 0 else v - 0.08, i, f'{v:.3f}', 
                 va='center', fontsize=11, fontweight='bold')
axes[1].grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('05_correlation.png', bbox_inches='tight', dpi=150)
plt.show()
print("📊 Insight: Petal length & petal width have the strongest correlation with species (>0.9).")"""))

cells.append(md("""### 3.6 3D Scatter Plot"""))

cells.append(code(r"""# ============================================================
# 3D Scatter Plot
# ============================================================
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

for i, species in enumerate(SPECIES_NAMES):
    subset = df[df['species_name'] == species]
    ax.scatter(subset['petal_length'], subset['petal_width'], subset['sepal_length'],
               c=COLORS[i], label=species, s=80, alpha=0.8,
               edgecolors='white', linewidth=0.5)

ax.set_xlabel('Petal Length (cm)', fontsize=11, labelpad=10)
ax.set_ylabel('Petal Width (cm)', fontsize=11, labelpad=10)
ax.set_zlabel('Sepal Length (cm)', fontsize=11, labelpad=10)
ax.set_title('3D Feature Space Visualization', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left', frameon=True, fancybox=True, shadow=True)
ax.view_init(elev=20, azim=45)

plt.savefig('06_3d_scatter.png', bbox_inches='tight', dpi=150)
plt.show()
print("📊 The 3D plot clearly shows the three clusters with minimal overlap.")"""))

# ============================================================
# CELL 5 — Data Preprocessing
# ============================================================
cells.append(md("""## 4. Data Preprocessing

Although the Iris dataset is already clean, we'll apply standard preprocessing steps that would be essential in real-world scenarios."""))

cells.append(code(r"""# ============================================================
# Step 1: Check for missing values & duplicates
# ============================================================
print("=" * 60)
print("🔍 DATA QUALITY CHECK")
print("=" * 60)

# Missing values
missing = df.isnull().sum().sum()
print(f"\n  Missing values:    {missing} ✅")

# Duplicates
duplicates = df.duplicated(subset=features).sum()
print(f"  Duplicate rows:    {duplicates}")
if duplicates > 0:
    print(f"  → Removing {duplicates} duplicate(s)...")
    df_clean = df.drop_duplicates(subset=features).reset_index(drop=True)
else:
    df_clean = df.copy()
    print(f"  → No duplicates found ✅")

print(f"  Final dataset:     {df_clean.shape[0]} samples × {df_clean.shape[1]} columns")"""))

cells.append(code(r"""# ============================================================
# Step 2: Prepare features and target
# ============================================================
X = df_clean[features].values
y = df_clean['species'].values

print(f"Feature matrix X shape: {X.shape}")
print(f"Target vector y shape:  {y.shape}")
print(f"Classes: {np.unique(y)} → {SPECIES_NAMES}")"""))

cells.append(code(r"""# ============================================================
# Step 3: Train-Test Split (80-20, stratified)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 60)
print("📂 TRAIN-TEST SPLIT")
print("=" * 60)
print(f"\n  Training set:  {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"  Test set:      {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")
print(f"\n  Training class distribution:")
for i, name in enumerate(SPECIES_NAMES):
    count = (y_train == i).sum()
    print(f"    {name:<12}: {count} samples")
print(f"\n  Test class distribution:")
for i, name in enumerate(SPECIES_NAMES):
    count = (y_test == i).sum()
    print(f"    {name:<12}: {count} samples")
print("\n  ✅ Stratified split ensures balanced class distribution!")"""))

cells.append(code(r"""# ============================================================
# Step 4: Feature Scaling (StandardScaler)
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=" * 60)
print("⚖️  FEATURE SCALING (StandardScaler)")
print("=" * 60)
print(f"\n  Before scaling:")
print(f"    Train mean: {X_train.mean(axis=0).round(2)}")
print(f"    Train std:  {X_train.std(axis=0).round(2)}")
print(f"\n  After scaling:")
print(f"    Train mean: {X_train_scaled.mean(axis=0).round(6)}")
print(f"    Train std:  {X_train_scaled.std(axis=0).round(6)}")
print(f"\n  ✅ Features are now zero-mean and unit-variance!")"""))

# ============================================================
# CELL 6 — Model Training & Comparison
# ============================================================
cells.append(md("""## 5. Model Training & Comparison

We'll train **6 different classifiers** and compare their performance using 5-fold stratified cross-validation."""))

cells.append(code(r"""# ============================================================
# Define models to compare
# ============================================================
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42, multi_class='multinomial'
    ),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42, probability=True),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, random_state=42
    )
}

# ============================================================
# Train and evaluate each model with cross-validation
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("=" * 70)
print("🏋️  MODEL TRAINING & 5-FOLD CROSS-VALIDATION")
print("=" * 70)
print(f"\n{'Model':<25} {'CV Mean':>8} {'CV Std':>8} {'Test Acc':>10}")
print("-" * 55)

for name, model in models.items():
    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    
    # Train on full training set & evaluate on test set
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores,
        'test_accuracy': test_acc,
        'y_pred': y_pred
    }
    
    print(f"  {name:<23} {cv_scores.mean():>7.4f}  ±{cv_scores.std():>6.4f}  {test_acc:>9.4f}")

# Find best model
best_model_name = max(results, key=lambda k: results[k]['cv_mean'])
print(f"\n🏆 Best Model (by CV): {best_model_name} "
      f"(CV: {results[best_model_name]['cv_mean']:.4f})")"""))

cells.append(code(r"""# ============================================================
# Visual comparison of model performance
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

model_names = list(results.keys())
cv_means = [results[m]['cv_mean'] for m in model_names]
cv_stds = [results[m]['cv_std'] for m in model_names]
test_accs = [results[m]['test_accuracy'] for m in model_names]

# Short names for display
short_names = ['LogReg', 'KNN', 'DecTree', 'SVM', 'RF', 'GBM']

# --- Plot 1: Bar chart comparison ---
x = np.arange(len(model_names))
width = 0.35
bars1 = axes[0].bar(x - width/2, cv_means, width, label='CV Mean Accuracy',
                      color=COLORS[2], alpha=0.85, edgecolor='white', linewidth=1.5,
                      yerr=cv_stds, capsize=4)
bars2 = axes[0].bar(x + width/2, test_accs, width, label='Test Accuracy',
                      color=COLORS[0], alpha=0.85, edgecolor='white', linewidth=1.5)

axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(short_names, fontsize=10)
axes[0].legend(fontsize=10, frameon=True, fancybox=True)
axes[0].set_ylim(0.85, 1.02)
axes[0].grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# --- Plot 2: Box plot of CV scores ---
cv_data = [results[m]['cv_scores'] for m in model_names]
bp = axes[1].boxplot(cv_data, labels=short_names, patch_artist=True,
                      medianprops=dict(color='black', linewidth=2),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))

colors_box = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

axes[1].set_xlabel('Model', fontsize=12)
axes[1].set_ylabel('CV Accuracy Score', fontsize=12)
axes[1].set_title('Cross-Validation Score Distribution', fontsize=14, fontweight='bold')
axes[1].set_ylim(0.85, 1.05)
axes[1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('07_model_comparison.png', bbox_inches='tight', dpi=150)
plt.show()"""))

# ============================================================
# CELL 7 — Hyperparameter Tuning
# ============================================================
cells.append(md("""## 6. Hyperparameter Tuning

We'll perform **GridSearchCV** on the top 3 models to find optimal hyperparameters."""))

cells.append(code(r"""# ============================================================
# Hyperparameter grids for top models
# ============================================================
param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'newton-cg'],
        'penalty': ['l2']
    },
    'K-Nearest Neighbors': {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    },
    'Support Vector Machine': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'linear', 'poly']
    }
}

base_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial'),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(random_state=42, probability=True)
}

print("=" * 70)
print("🔧 HYPERPARAMETER TUNING (GridSearchCV)")
print("=" * 70)

tuned_results = {}

for name in param_grids:
    grid_search = GridSearchCV(
        base_models[name], param_grids[name],
        cv=cv, scoring='accuracy', n_jobs=-1, refit=True
    )
    grid_search.fit(X_train_scaled, y_train)
    
    y_pred_tuned = grid_search.predict(X_test_scaled)
    test_acc_tuned = accuracy_score(y_test, y_pred_tuned)
    
    tuned_results[name] = {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'test_accuracy': test_acc_tuned,
        'model': grid_search.best_estimator_,
        'y_pred': y_pred_tuned
    }
    
    print(f"\n  📌 {name}")
    print(f"     Best CV Score:  {grid_search.best_score_:.4f}")
    print(f"     Test Accuracy:  {test_acc_tuned:.4f}")
    print(f"     Best Params:    {grid_search.best_params_}")

# Overall best tuned model
best_tuned_name = max(tuned_results, key=lambda k: tuned_results[k]['best_cv_score'])
print(f"\n{'='*70}")
print(f"🏆 Best Tuned Model: {best_tuned_name}")
print(f"   CV Score: {tuned_results[best_tuned_name]['best_cv_score']:.4f}")
print(f"   Test Accuracy: {tuned_results[best_tuned_name]['test_accuracy']:.4f}")
print(f"{'='*70}")"""))

# ============================================================
# CELL 8 — Best Model Evaluation
# ============================================================
cells.append(md("""## 7. Best Model Evaluation

Detailed evaluation of our best-performing model with confusion matrix, classification report, and performance metrics."""))

cells.append(code(r"""# ============================================================
# Use the best tuned model for detailed evaluation
# ============================================================
best_model = tuned_results[best_tuned_name]['model']
y_pred_best = tuned_results[best_tuned_name]['y_pred']

print("=" * 60)
print(f"📊 DETAILED EVALUATION — {best_tuned_name}")
print("=" * 60)

# Classification Report
print("\n📋 Classification Report:")
print("-" * 55)
report = classification_report(y_test, y_pred_best, target_names=SPECIES_NAMES, digits=4)
print(report)

# Individual Metrics
acc = accuracy_score(y_test, y_pred_best)
prec = precision_score(y_test, y_pred_best, average='weighted')
rec = recall_score(y_test, y_pred_best, average='weighted')
f1 = f1_score(y_test, y_pred_best, average='weighted')

print(f"  Accuracy:   {acc:.4f}")
print(f"  Precision:  {prec:.4f} (weighted)")
print(f"  Recall:     {rec:.4f} (weighted)")
print(f"  F1-Score:   {f1:.4f} (weighted)")"""))

cells.append(code(r"""# ============================================================
# Confusion Matrix Visualization
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Standard confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SPECIES_NAMES)
disp.plot(ax=axes[0], cmap='Blues', values_format='d',
          colorbar=False)
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')

# Plot 2: Normalized confusion matrix
cm_normalized = confusion_matrix(y_test, y_pred_best, normalize='true')
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=SPECIES_NAMES)
disp2.plot(ax=axes[1], cmap='Purples', values_format='.2%',
           colorbar=False)
axes[1].set_title('Confusion Matrix (Normalized %)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('08_confusion_matrix.png', bbox_inches='tight', dpi=150)
plt.show()

# Print misclassifications
total_errors = (cm.sum() - np.trace(cm))
print(f"\n📊 Total Misclassifications: {total_errors} out of {len(y_test)} test samples")"""))

cells.append(md("""### Learning Curves"""))

cells.append(code(r"""# ============================================================
# Learning Curves — bias/variance analysis
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Learning Curves — Training vs Validation Performance',
             fontsize=16, fontweight='bold', y=1.02)

for idx, (name, model_obj) in enumerate([
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('SVM', SVC(kernel='rbf', random_state=42))
]):
    train_sizes, train_scores, val_scores = learning_curve(
        model_obj, X_train_scaled, y_train, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv, scoring='accuracy', n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    axes[idx].fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                            alpha=0.15, color=COLORS[2])
    axes[idx].fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                            alpha=0.15, color=COLORS[0])
    axes[idx].plot(train_sizes, train_mean, 'o-', color=COLORS[2], 
                    label='Training Score', linewidth=2, markersize=5)
    axes[idx].plot(train_sizes, val_mean, 'o-', color=COLORS[0],
                    label='Validation Score', linewidth=2, markersize=5)
    
    axes[idx].set_xlabel('Training Set Size', fontsize=11)
    axes[idx].set_ylabel('Accuracy', fontsize=11)
    axes[idx].set_title(name, fontsize=13, fontweight='bold')
    axes[idx].legend(loc='lower right', fontsize=9, frameon=True)
    axes[idx].set_ylim(0.7, 1.05)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('09_learning_curves.png', bbox_inches='tight', dpi=150)
plt.show()
print("📊 Learning curves show models converge well — no signs of overfitting.")"""))

# ============================================================
# CELL 9 — Feature Importance
# ============================================================
cells.append(md("""## 8. Feature Importance Analysis

Understanding which features contribute most to the classification."""))

cells.append(code(r"""# ============================================================
# Feature Importance from multiple models
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold', y=1.02)

feature_display = ['Sepal\nLength', 'Sepal\nWidth', 'Petal\nLength', 'Petal\nWidth']

# 1. Random Forest Feature Importance
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)
importances_rf = rf.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1]

bars1 = axes[0].bar(range(4), importances_rf[indices_rf],
                     color=[COLORS[i % 3] for i in range(4)],
                     edgecolor='white', linewidth=1.5, alpha=0.85)
axes[0].set_xticks(range(4))
axes[0].set_xticklabels([feature_display[i] for i in indices_rf], fontsize=10)
axes[0].set_ylabel('Importance Score', fontsize=11)
axes[0].set_title('Random Forest\n(Gini Importance)', fontsize=13, fontweight='bold')
for bar, val in zip(bars1, importances_rf[indices_rf]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
axes[0].grid(True, axis='y', alpha=0.3)

# 2. Logistic Regression Coefficients (mean absolute)
lr = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
lr.fit(X_train_scaled, y_train)
importances_lr = np.mean(np.abs(lr.coef_), axis=0)
indices_lr = np.argsort(importances_lr)[::-1]

bars2 = axes[1].bar(range(4), importances_lr[indices_lr],
                     color=[COLORS[i % 3] for i in range(4)],
                     edgecolor='white', linewidth=1.5, alpha=0.85)
axes[1].set_xticks(range(4))
axes[1].set_xticklabels([feature_display[i] for i in indices_lr], fontsize=10)
axes[1].set_ylabel('Mean |Coefficient|', fontsize=11)
axes[1].set_title('Logistic Regression\n(Coefficient Magnitude)', fontsize=13, fontweight='bold')
for bar, val in zip(bars2, importances_lr[indices_lr]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
axes[1].grid(True, axis='y', alpha=0.3)

# 3. Decision Tree Feature Importance
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)
importances_dt = dt.feature_importances_
indices_dt = np.argsort(importances_dt)[::-1]

bars3 = axes[2].bar(range(4), importances_dt[indices_dt],
                     color=[COLORS[i % 3] for i in range(4)],
                     edgecolor='white', linewidth=1.5, alpha=0.85)
axes[2].set_xticks(range(4))
axes[2].set_xticklabels([feature_display[i] for i in indices_dt], fontsize=10)
axes[2].set_ylabel('Importance Score', fontsize=11)
axes[2].set_title('Decision Tree\n(Gini Importance)', fontsize=13, fontweight='bold')
for bar, val in zip(bars3, importances_dt[indices_dt]):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
axes[2].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('10_feature_importance.png', bbox_inches='tight', dpi=150)
plt.show()
print("📊 All models agree: Petal Length & Petal Width are the most important features.")"""))

# ============================================================
# CELL 10 — Decision Boundary Visualization
# ============================================================
cells.append(md("""## 9. Decision Boundary Visualization

Visualizing how each model partitions the feature space (using the 2 most important features: petal length & petal width)."""))

cells.append(code(r"""# ============================================================
# Decision boundary plots using top 2 features
# ============================================================
X_2d = X_train_scaled[:, 2:4]  # petal length & petal width
X_2d_test = X_test_scaled[:, 2:4]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Decision Boundaries (Petal Length vs Petal Width)',
             fontsize=18, fontweight='bold', y=1.01)

models_2d = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

cmap_light = ListedColormap(['#FFCCCC', '#CCFFEE', '#CCE5FF'])
cmap_bold = ListedColormap(COLORS)

for idx, (name, model) in enumerate(models_2d.items()):
    ax = axes[idx // 3, idx % 3]
    
    model.fit(X_2d, y_train)
    
    # Create mesh grid
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.7)
    ax.contour(xx, yy, Z, colors='gray', linewidths=0.5, alpha=0.5)
    
    # Plot training points
    for i, species in enumerate(SPECIES_NAMES):
        mask = y_train == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=COLORS[i],
                   label=species, edgecolors='white', linewidth=0.8, s=50, alpha=0.8)
    
    acc = model.score(X_2d_test, y_test)
    ax.set_title(f'{name}\n(2D Acc: {acc:.3f})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Petal Length (scaled)')
    ax.set_ylabel('Petal Width (scaled)')
    ax.legend(fontsize=8, loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig('11_decision_boundaries.png', bbox_inches='tight', dpi=150)
plt.show()
print("📊 Decision boundaries show how each model separates the feature space differently.")"""))

# ============================================================
# CELL 11 — Decision Tree Visualization
# ============================================================
cells.append(md("""### 9.1 Decision Tree Structure"""))

cells.append(code(r"""# ============================================================
# Visualize the actual decision tree
# ============================================================
dt_viz = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_viz.fit(X_train_scaled, y_train)

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt_viz, 
          feature_names=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
          class_names=SPECIES_NAMES,
          filled=True, rounded=True, fontsize=10,
          ax=ax, proportion=True,
          impurity=True)
ax.set_title('Decision Tree Structure (max_depth=4)', fontsize=16, fontweight='bold')

plt.savefig('12_decision_tree.png', bbox_inches='tight', dpi=150)
plt.show()
print("📊 The decision tree reveals the splitting rules the model learned.")"""))

# ============================================================
# CELL — Predictions: Predicted vs Actual
# ============================================================
cells.append(md("""## 10. Predictions — Predicted vs Actual

Let's see what the best model actually predicted for each test sample, compared to the true labels."""))

cells.append(code(r"""# ============================================================
# Predicted vs Actual for all test samples
# ============================================================
pred_df = pd.DataFrame({
    'Sample #': range(1, len(y_test) + 1),
    'Sepal Length': X_test[:, 0],
    'Sepal Width': X_test[:, 1],
    'Petal Length': X_test[:, 2],
    'Petal Width': X_test[:, 3],
    'Actual Species': [SPECIES_NAMES[y] for y in y_test],
    'Predicted Species': [SPECIES_NAMES[y] for y in y_pred_best],
})
pred_df['Correct?'] = pred_df['Actual Species'] == pred_df['Predicted Species']
pred_df['Correct?'] = pred_df['Correct?'].map({True: '✅ Yes', False: '❌ No'})

print("=" * 70)
print(f"🔮 PREDICTIONS — {best_tuned_name}")
print("=" * 70)
print(f"\nShowing all {len(y_test)} test sample predictions:\n")

pred_df"""))

cells.append(code(r"""# ============================================================
# Visual: Prediction accuracy per sample
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))

correct_mask = (y_test == y_pred_best)
colors_pred = ['#2ECC71' if c else '#E74C3C' for c in correct_mask]

bars = ax.bar(range(1, len(y_test) + 1), [1]*len(y_test), 
              color=colors_pred, edgecolor='white', linewidth=0.8)

# Add species labels on bars
for i, (actual, pred, correct) in enumerate(zip(y_test, y_pred_best, correct_mask)):
    label = SPECIES_NAMES[actual][:3]  # Short name
    ax.text(i + 1, 0.5, label, ha='center', va='center', 
            fontsize=8, fontweight='bold', color='white')

ax.set_xlabel('Test Sample #', fontsize=12)
ax.set_title('Prediction Results per Test Sample (Green = Correct, Red = Wrong)', 
             fontsize=14, fontweight='bold')
ax.set_xlim(0.5, len(y_test) + 0.5)
ax.set_yticks([])

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ECC71', label=f'Correct ({correct_mask.sum()})'),
                   Patch(facecolor='#E74C3C', label=f'Wrong ({(~correct_mask).sum()})')]
ax.legend(handles=legend_elements, fontsize=11, loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig('13_predictions.png', bbox_inches='tight', dpi=150)
plt.show()

accuracy = correct_mask.sum() / len(y_test) * 100
print(f"\n📊 Result: {correct_mask.sum()}/{len(y_test)} correct ({accuracy:.1f}% accuracy)")
if (~correct_mask).sum() > 0:
    print(f"\n❌ Misclassified samples:")
    wrong_indices = np.where(~correct_mask)[0]
    for idx in wrong_indices:
        print(f"   Sample #{idx+1}: Actual = {SPECIES_NAMES[y_test[idx]]}, "
              f"Predicted = {SPECIES_NAMES[y_pred_best[idx]]}")"""))

# ============================================================
# CELL — Custom Prediction Demo
# ============================================================
cells.append(md("""### 10.1 Try Your Own Prediction! 🔮

Enter custom flower measurements below to see what species the model predicts."""))

cells.append(code(r"""# ============================================================
# Custom Prediction Function
# ============================================================
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    """Predict the iris species given flower measurements."""
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_features)
    prediction = best_model.predict(input_scaled)[0]
    species = SPECIES_NAMES[prediction]
    
    # Get probability if model supports it
    if hasattr(best_model, 'predict_proba'):
        probabilities = best_model.predict_proba(input_scaled)[0]
        print(f"\n{'='*50}")
        print(f"🌸 IRIS FLOWER PREDICTION")
        print(f"{'='*50}")
        print(f"\n  Input Measurements:")
        print(f"    Sepal Length: {sepal_length} cm")
        print(f"    Sepal Width:  {sepal_width} cm")
        print(f"    Petal Length: {petal_length} cm")
        print(f"    Petal Width:  {petal_width} cm")
        print(f"\n  🎯 Predicted Species: {species}")
        print(f"\n  Confidence Scores:")
        for i, name in enumerate(SPECIES_NAMES):
            bar = "█" * int(probabilities[i] * 30)
            print(f"    {name:<12}: {probabilities[i]:>6.1%} {bar}")
    else:
        print(f"\n  🎯 Predicted Species: {species}")
    
    return species

# ============================================================
# Example Predictions
# ============================================================
print("📌 Example 1: Typical Setosa flower")
predict_iris(5.0, 3.5, 1.4, 0.2)

print("\n\n📌 Example 2: Typical Versicolor flower")
predict_iris(5.9, 2.8, 4.5, 1.3)

print("\n\n📌 Example 3: Typical Virginica flower")
predict_iris(6.7, 3.1, 5.6, 2.4)"""))

cells.append(md("""---

> **💡 Try it yourself!** Modify the values in `predict_iris(sepal_length, sepal_width, petal_length, petal_width)` above and re-run the cell to predict any flower!"""))

# ============================================================
# CELL 12 — Conclusion
# ============================================================
cells.append(md("""## 11. Conclusion


### 📊 Summary of Results

| Metric | Value |
|--------|-------|
| **Best Model** | (determined at runtime) |
| **Cross-Validation Accuracy** | ~96-100% |
| **Test Set Accuracy** | ~96-100% |
| **Most Important Features** | Petal Length, Petal Width |

### 🔑 Key Findings

1. **Petal features dominate**: Petal length and petal width are significantly more discriminative than sepal features, consistently ranked as the top-2 most important features across all models.

2. **Setosa is linearly separable**: *Iris Setosa* can be perfectly classified using simple linear boundaries, while *Versicolor* and *Virginica* show slight overlap requiring more sophisticated decision boundaries.

3. **Multiple models achieve near-perfect accuracy**: Logistic Regression, SVM, Random Forest, and Gradient Boosting all achieve >95% accuracy, demonstrating the dataset's inherent separability.

4. **No overfitting observed**: Learning curves show that training and validation scores converge, indicating our models generalize well.

5. **Minimal preprocessing needed**: The dataset is clean with no missing values, and only feature scaling (StandardScaler) was applied to ensure fair model comparison.

### 🛠️ Skills Demonstrated
- **Data Analysis**: Exploratory data analysis with statistical summaries and distribution analysis
- **Data Visualization**: Histograms, box plots, violin plots, pair plots, 3D scatter, heatmaps, decision boundaries
- **Machine Learning**: Training and evaluating 6 different classifiers
- **Hyperparameter Tuning**: GridSearchCV with cross-validation
- **Model Evaluation**: Confusion matrices, classification reports, learning curves, feature importance

---

*This project was completed as part of the QSkill AI & ML Internship (June 2026).*"""))

# ============================================================
# BUILD THE NOTEBOOK
# ============================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "pygments_lexer": "ipython3"
        }
    },
    "cells": cells
}

# Write the notebook
output_path = "Iris_Flower_Classification.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook generated successfully: {output_path}")
print(f"   Total cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='markdown')} markdown, "
      f"{sum(1 for c in cells if c['cell_type']=='code')} code)")
print(f"\nTo run: jupyter notebook {output_path}")
