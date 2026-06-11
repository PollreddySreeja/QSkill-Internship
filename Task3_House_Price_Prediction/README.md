# 🏡 House Price Prediction System 

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)

An industry-grade machine learning web application that predicts property prices based on housing features (Area, Bedrooms, Bathrooms, Location Score). Built as part of **QSkill Task 3**, this project goes beyond a simple assignment by providing a full-stack, visually rich experience with an end-to-end ML pipeline.

---

## 🚀 Project Overview

**What problem is being solved?**  
House price prediction helps buyers, sellers, and real estate companies estimate property values using machine learning models trained on historical housing data. 

Rather than just printing a single MSE score in a notebook, this project provides a **full web interface** allowing users to interact with the model directly and view detailed data analytics.

### 🌟 Key Features
- **Live Prediction Interface:** A clean, responsive form with asynchronous (AJAX) prediction handling and loading animations.
- **Data Distribution Analytics:** Interactive visual analysis of housing distributions (Price, Age, Rooms, Bedrooms).
- **Advanced Insights:** 
  - **Correlation Heatmap** showing feature relationships.
  - **Actual vs Predicted Graph** validating model accuracy.
  - **Feature Importance** breakdown of the Linear Regression coefficients.
- **Transparent Workflow:** Clear display of the preprocessing steps (Missing Value Checks, Feature Scaling) and the 80/20 Train-Test split.

---

## 📊 Dataset Information

This model is trained on the standard **California Housing Dataset** (used as the modern replacement for the deprecated Boston Housing Dataset).

- **Total Records:** 20,640
- **Features:** 8 Numerical
- **Target Variable:** Median House Value
- **Missing Values:** 0

### ML Workflow
`Dataset Loading` ➔ `Data Cleaning` ➔ `Feature Scaling (StandardScaler)` ➔ `Train/Test Split (80/20)` ➔ `Linear Regression Training` ➔ `Evaluation (MSE/RMSE/R²)`

---

## 🛠️ Technology Stack

**Backend & Machine Learning:**
- Python
- Pandas & NumPy
- Scikit-Learn (Linear Regression)
- Flask (Web Server)

**Frontend:**
- HTML5 & CSS3 (Custom styling)
- Vanilla JavaScript (AJAX functionality)
- FontAwesome (Icons)

---

## 💻 How to Run Locally

Follow these steps to set up the project on your local machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PollreddySreeja/QSkill-Internship.git
   cd QSkill-Internship/Task3_House_Price_Prediction
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model & Generate Visualizations:**
   Run the ML pipeline script to generate the `model.pkl`, `scaler.pkl`, and plot images inside `static/images/`.
   ```bash
   python train_model.py
   ```

4. **Start the Flask Web Server:**
   ```bash
   python app.py
   ```

5. **Access the Application:**
   Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 🔮 Future Enhancements
- Upgrade model to **Random Forest Regression** or **XGBoost** for higher accuracy.
- Integrate a live **Real Estate API** to fetch current market rates.
- Deploy the web application via Docker/Heroku.

---

*Developed for QSkill Internship Task 3.*
