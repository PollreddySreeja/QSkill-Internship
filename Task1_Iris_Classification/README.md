<div align="center">

# Iris Flower Classification

**QSkill AI & ML Internship · Task 1**

Classify iris flowers into three species — *Setosa*, *Versicolor*, and *Virginica* — using sepal and petal measurements.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()

</div>

<br>

## Overview

This project implements a complete machine learning pipeline on the classic **Iris dataset** (150 samples, 4 features, 3 classes). Six different classifiers are trained, tuned, and evaluated with comprehensive visualizations.

<br>

## Visualizations

<p align="center">
  <img src="04_pair_plot.png" width="45%" />
  <img src="05_correlation.png" width="45%" />
</p>
<p align="center">
  <img src="08_confusion_matrix.png" width="45%" />
  <img src="07_model_comparison.png" width="45%" />
</p>
<p align="center">
  <img src="11_decision_boundaries.png" width="45%" />
  <img src="10_feature_importance.png" width="45%" />
</p>

<br>

## Pipeline

```
Load Data → EDA → Preprocess → Train/Test Split → Feature Scaling
                                                        ↓
                                        Train 6 Models (LR, KNN, DT, SVM, RF, GB)
                                                        ↓
                                        5-Fold Cross Validation
                                                        ↓
                                        Hyperparameter Tuning (GridSearchCV)
                                                        ↓
                                        Evaluate & Visualize Results
```

<br>

## Models & Results

All models achieve **>95% test accuracy**.

| Model | Description |
|:------|:------------|
| Logistic Regression | Linear classifier with multinomial output |
| K-Nearest Neighbors | Distance-based instance learning (k=5) |
| Decision Tree | Tree-based splits using Gini impurity |
| Support Vector Machine | RBF kernel with soft margins |
| Random Forest | Ensemble of 100 decision trees |
| Gradient Boosting | Sequential boosted ensemble |

**Key findings:**
- Petal length and petal width are the most discriminative features
- Iris Setosa is linearly separable from the other two species
- No overfitting observed across any model

<br>

## Setup

```bash
pip install -r requirements.txt
jupyter notebook Iris_Flower_Classification.ipynb
```

Then run **Kernel → Restart & Run All**.

<br>

## Files

```
├── Iris_Flower_Classification.ipynb   # Full ML pipeline notebook
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── generate_notebook.py               # Notebook generator script
└── *.png                              # 12 output visualizations
```

<br>

## Skills Demonstrated

- Data exploration and statistical analysis
- 12+ visualization types (distributions, pair plots, heatmaps, 3D scatter, decision boundaries)
- Multi-model training and comparison
- Hyperparameter tuning with GridSearchCV
- Model evaluation (accuracy, precision, recall, F1, confusion matrices)
- Feature importance analysis

<br>

---

<div align="center">

[← Back to Main Repo](../README.md)

</div>
