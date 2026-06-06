<div align="center">

# Iris Flower Classification

**QSkill AI & ML Internship · Task 1**

Classify iris flowers into three species — *Setosa*, *Versicolor*, and *Virginica* — based on measurements of their petals and sepals.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()

</div>

<br>

## Objective

Build a classifier that distinguishes between three iris flower species based on four physical measurements (sepal length, sepal width, petal length, petal width).

<br>

## Dataset

The classic **Iris dataset** from scikit-learn (originally from the UCI Machine Learning Repository, Fisher 1936).

- **150 samples** · **4 features** · **3 classes** (50 per class)
- Perfectly balanced, no missing values

| Feature | Description |
|:--------|:------------|
| Sepal Length | Length of the sepal in cm |
| Sepal Width | Width of the sepal in cm |
| Petal Length | Length of the petal in cm |
| Petal Width | Width of the petal in cm |

<br>

---

## Step 1 · Load the dataset and explore it visually

Loaded the Iris dataset using `sklearn.datasets.load_iris()` and converted it into a Pandas DataFrame for analysis.

```python
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
```

Explored the data using:
- `df.head()`, `df.info()`, `df.describe()` — statistical summaries
- Checked class balance — 50 samples per species (perfectly balanced)

**Visual exploration performed:**

- **Histograms with KDE curves** — distribution of each feature per species
- **Box plots** — outlier detection across species
- **Violin plots** — distribution shape and density visualization
- **Pair plot** — pairwise scatter plots of all feature combinations
- **Correlation heatmap** — feature-to-feature and feature-to-target correlation
- **3D scatter plot** — multi-dimensional cluster separation

<p align="center">
  <img src="01_feature_distributions.png" width="45%" />
  <img src="04_pair_plot.png" width="45%" />
</p>
<p align="center">
  <img src="02_box_plots.png" width="45%" />
  <img src="06_3d_scatter.png" width="45%" />
</p>

**Key insight:** Petal features show much clearer species separation than sepal features. Setosa is linearly separable.

<br>

---

## Step 2 · Split the data into training/test sets

Used **stratified sampling** to maintain equal class distribution in both sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- **Training set:** 120 samples (80%)
- **Test set:** 30 samples (20%)
- Stratified split ensures each species is proportionally represented in both sets

<br>

---

## Step 3 · Preprocess if needed

Although the Iris dataset is clean, we applied standard preprocessing steps:

```python
# Check for missing values
missing = df.isnull().sum().sum()    # Result: 0

# Check for duplicates
duplicates = df.duplicated().sum()   # Removed if any found

# Feature scaling — zero mean, unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   # Only transform, never fit on test!
```

| Check | Result |
|:------|:-------|
| Missing values | 0 — no missing data |
| Duplicates | Removed if found |
| Scaling | StandardScaler applied (mean=0, std=1) |

**Why scale?** KNN and SVM are distance-based algorithms — features with larger ranges would dominate without scaling.

<p align="center">
  <img src="05_correlation.png" width="45%" />
  <img src="03_violin_plots.png" width="45%" />
</p>

<br>

---

## Step 4 · Train a simple classifier

Trained **6 different classifiers** on the scaled training data and compared them using **5-fold stratified cross-validation**.

```python
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='multinomial'),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree':       DecisionTreeClassifier(),
    'Support Vector Machine': SVC(kernel='rbf', probability=True),
    'Random Forest':       RandomForestClassifier(n_estimators=100),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100)
}

# 5-fold stratified cross-validation for each model
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    model.fit(X_train_scaled, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
```

### Hyperparameter Tuning

Applied **GridSearchCV** on the top 3 models to find optimal hyperparameters:

```python
# Example: KNN tuning
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
```

Models tuned with GridSearchCV:
- **Logistic Regression** — tuned `C`, `solver`, `penalty`
- **K-Nearest Neighbors** — tuned `n_neighbors`, `weights`, `metric`
- **Support Vector Machine** — tuned `C`, `gamma`, `kernel`

<p align="center">
  <img src="07_model_comparison.png" width="45%" />
  <img src="12_decision_tree.png" width="45%" />
</p>

<br>

---

## Step 5 · Evaluate with accuracy, precision, or confusion matrix

Evaluated all models using multiple metrics:

```python
# Detailed evaluation
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec  = recall_score(y_test, y_pred, average='weighted')
f1   = f1_score(y_test, y_pred, average='weighted')
cm   = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred, target_names=['Setosa', 'Versicolor', 'Virginica']))
```

### Results

All models achieve **>95% accuracy** on the test set.

### Evaluation outputs generated:
- **Classification report** — precision, recall, F1-score per species
- **Confusion matrix** — counts + normalized percentage heatmaps
- **Learning curves** — training vs validation accuracy to check overfitting
- **Feature importance** — ranked by 3 methods (Random Forest, Gradient Boosting, Logistic Regression coefficients)
- **Decision boundaries** — 2D visualization of how each model separates species

<p align="center">
  <img src="08_confusion_matrix.png" width="45%" />
  <img src="10_feature_importance.png" width="45%" />
</p>
<p align="center">
  <img src="09_learning_curves.png" width="45%" />
  <img src="11_decision_boundaries.png" width="45%" />
</p>

### Key Findings

- **Petal length** and **petal width** are the most important features (importance > 0.4)
- **Iris Setosa** is perfectly classified by all models — linearly separable
- **Versicolor & Virginica** have slight overlap but are well-separated by ensemble methods
- **No overfitting** — learning curves show training and validation scores converge
- **Hyperparameter tuning** provides marginal improvement on already-strong baselines

<br>

---

## Skills Gained

- **Numeric data analysis** — statistical summaries, distribution analysis, correlation study
- **Classification modeling** — training and comparing 6 ML algorithms with cross-validation
- **Evaluating results** — accuracy, precision, recall, F1-score, confusion matrices, learning curves

<br>

## Setup & Run

```bash
pip install -r requirements.txt
jupyter notebook Iris_Flower_Classification.ipynb
```

Then click **Kernel → Restart & Run All** to execute the full pipeline.

<br>

## Files

```
├── Iris_Flower_Classification.ipynb   # Full ML pipeline notebook
├── README.md                          # This file
├── requirements.txt                   # Dependencies (numpy, pandas, matplotlib, seaborn, scikit-learn)
├── generate_notebook.py               # Script to regenerate the notebook
└── *.png                              # 12 output visualizations
```

<br>

---

<div align="center">

[← Back to Main Repo](../README.md)

</div>

