<![CDATA[<div align="center">

# 🌸 Iris Flower Classification

### QSkill AI & ML Internship — Task 1

> *Classify iris flowers into three species (Setosa, Versicolor, Virginica) based on sepal and petal measurements using multiple machine learning algorithms.*

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)]()

</div>

---

## 📋 Table of Contents
- [About the Project](#-about-the-project)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Models Compared](#-models-compared)
- [Key Results](#-key-results)
- [Visualizations](#-visualizations)
- [Setup & Installation](#️-setup--installation)
- [Project Structure](#-project-structure)
- [Skills Demonstrated](#️-skills-demonstrated)

---

## 📖 About the Project

This project is part of the **QSkill AI & Machine Learning Internship (June 2026)**. The objective is to build a robust classifier that accurately identifies iris flower species based on four physical measurements.

### ML Pipeline Implemented

```
📥 Data Loading ──→ 🔍 Exploratory Data Analysis ──→ 🔧 Preprocessing
                                                           │
                    ┌──────────────────────────────────────┘
                    ▼
              📐 Feature Scaling ──→ 🤖 Model Training (6 classifiers)
                                           │
                                           ▼
                                     📊 Cross-Validation (5-fold stratified)
                                           │
                                           ▼
                                     ⚡ Hyperparameter Tuning (GridSearchCV)
                                           │
                                           ▼
                                     ✅ Evaluation & Analysis
```

---

## 📊 Dataset

The **Iris dataset** (Fisher, 1936) is one of the most well-known datasets in pattern recognition and machine learning.

| Property | Details |
|:---------|:--------|
| **Samples** | 150 |
| **Features** | 4 — sepal length, sepal width, petal length, petal width |
| **Target Classes** | 3 — *Setosa*, *Versicolor*, *Virginica* |
| **Class Balance** | Perfectly balanced (50 samples per class) |
| **Missing Values** | None |
| **Source** | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/iris) / scikit-learn built-in |

---

## 🔬 Methodology

### 1️⃣ Data Loading & Exploration
- Loaded dataset from `sklearn.datasets`
- Generated statistical summaries (`describe()`, `info()`)
- Verified data integrity — no missing values, minimal duplicates

### 2️⃣ Exploratory Data Analysis (EDA)
- **10+ visualizations** covering distributions, correlations, and inter-feature relationships
- Key insight: Petal measurements are far more discriminative than sepal measurements

### 3️⃣ Data Preprocessing
- Duplicate removal
- Feature scaling using **StandardScaler** (zero mean, unit variance)
- Stratified 80/20 train-test split

### 4️⃣ Model Training & Comparison
- Trained **6 different classifiers** on the same preprocessed data
- Used **5-fold stratified cross-validation** for robust evaluation

### 5️⃣ Hyperparameter Tuning
- Applied **GridSearchCV** for systematic hyperparameter optimization
- Compared tuned vs. baseline performance

### 6️⃣ Comprehensive Evaluation
- Accuracy, precision, recall, F1-score per class
- Confusion matrices (absolute + normalized)
- Learning curves for overfitting analysis
- Multi-method feature importance analysis

---

## 🤖 Models Compared

| Model | Description | Key Hyperparameters |
|:------|:------------|:-------------------|
| **Logistic Regression** | Linear model with multinomial classification | `C`, `solver`, `max_iter` |
| **K-Nearest Neighbors** | Instance-based learning | `n_neighbors`, `weights`, `metric` |
| **Decision Tree** | Tree-based model with Gini criterion | `max_depth`, `min_samples_split` |
| **Support Vector Machine** | RBF kernel SVM | `C`, `gamma`, `kernel` |
| **Random Forest** | Ensemble of 100 decision trees | `n_estimators`, `max_depth` |
| **Gradient Boosting** | Sequential ensemble with boosting | `n_estimators`, `learning_rate` |

---

## 📈 Key Results

<div align="center">

### All models achieve **>95% accuracy** on the test set

</div>

#### Key Findings:
- 🎯 **Petal features** (length & width) are significantly more discriminative than sepal features
- 📐 **Iris Setosa** is linearly separable from the other two species
- ✅ **No overfitting** observed — learning curves show proper convergence
- ⚡ **Hyperparameter tuning** provides marginal improvement on already-strong baselines

---

## 📸 Visualizations

The notebook generates **12 publication-quality plots**:

| # | Visualization | Purpose |
|:-:|:-------------|:--------|
| 1 | Feature Distributions (KDE) | Understand spread of each feature per species |
| 2 | Box Plots | Detect outliers and compare medians |
| 3 | Violin Plots | Visualize distribution shapes and density |
| 4 | Pair Plot | Explore pairwise feature relationships |
| 5 | Correlation Heatmap | Identify feature correlations |
| 6 | 3D Scatter Plot | Multi-dimensional species separation |
| 7 | Model Comparison Bar Chart | Side-by-side accuracy comparison |
| 8 | Confusion Matrices | Prediction error analysis (counts + normalized) |
| 9 | Learning Curves | Detect overfitting/underfitting |
| 10 | Feature Importance | Multi-model importance ranking |
| 11 | Decision Boundaries | 2D classification regions for all 6 models |
| 12 | Decision Tree Structure | Visual tree representation |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### Install & Run

```bash
# Clone the repo
git clone https://github.com/PollreddySreeja/QSkill-Internship.git
cd QSkill-Internship/Task1_Iris_Classification

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook Iris_Flower_Classification.ipynb
```

Then click **Kernel → Restart & Run All** to execute all cells.

---

## 📁 Project Structure

```
Task1_Iris_Classification/
├── 📓 Iris_Flower_Classification.ipynb   # Main Jupyter notebook (full pipeline)
├── 📖 README.md                          # This file
├── 📦 requirements.txt                   # Python dependencies
├── 🐍 generate_notebook.py               # Script to regenerate the notebook
├── 📊 01_feature_distributions.png       # Distribution plots
├── 📊 02_box_plots.png                   # Box plots
├── 📊 03_violin_plots.png                # Violin plots
├── 📊 04_pair_plot.png                   # Pair plot
├── 📊 05_correlation.png                 # Correlation heatmap
├── 📊 06_3d_scatter.png                  # 3D scatter plot
├── 📊 07_model_comparison.png            # Model comparison
├── 📊 08_confusion_matrix.png            # Confusion matrices
├── 📊 09_learning_curves.png             # Learning curves
├── 📊 10_feature_importance.png          # Feature importance
├── 📊 11_decision_boundaries.png         # Decision boundaries
└── 📊 12_decision_tree.png               # Decision tree visualization
```

---

## 🛠️ Skills Demonstrated

- ✅ **Numeric Data Analysis** — Statistical summaries, distribution analysis, outlier detection
- ✅ **Data Visualization** — 12+ professional plot types with Matplotlib & Seaborn
- ✅ **Classification Modeling** — 6 different ML algorithms trained & compared
- ✅ **Model Evaluation** — Accuracy, precision, recall, F1-score, confusion matrices
- ✅ **Hyperparameter Tuning** — GridSearchCV with stratified cross-validation
- ✅ **Feature Engineering** — StandardScaler, multi-method feature importance analysis

---

<div align="center">

**[⬆ Back to Main Repo](../README.md)**

*Made with ❤️ by [Pollreddy Sreeja](https://github.com/PollreddySreeja) for QSkill Internship*

</div>
]]>
