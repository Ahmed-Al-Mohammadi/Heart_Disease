-----

# 🫀 Heart Disease Prediction Project

[](https://www.python.org/)
[](https://streamlit.io)
[](https://scikit-learn.org/)
[](https://opensource.org/licenses/MIT)

An end-to-end machine learning project to analyze heart patient data, train a model to predict the likelihood of heart disease based on medical attributes, and deploy it as an interactive Streamlit web application.

## 🚀 Live Demo App

This interactive application, built with `Streamlit`, allows a user to input medical data and receive an immediate prediction about the likelihood of having heart disease.

*(It is highly recommended to add a screenshot or GIF of your Streamlit app in action here)*

## ✨ Features

  * **Exploratory Data Analysis (EDA):** Comprehensive and automated data analysis using `ydata-profiling` and visualizations with `seaborn`/`matplotlib` (detailed in `Heart_Disease.ipynb`).
  * **Multiple Model Training:** Comparison of several classic machine learning models (e.g., `LogisticRegression`, `RandomForest`, `SVC`, `KNeighborsClassifier`) to select the best performer.
  * **Hyperparameter Tuning:** Use of `GridSearchCV` to fine-tune the best-performing model (Random Forest).
  * **Model Explainability (XAI):** Implementation of `shap` to understand the most important features influencing the model's predictions.
  * **Interactive Web App:** A simple and direct user interface built with `Streamlit` for patient data input and immediate prediction output with a confidence score.

## 🛠️ Methodology (Pipeline)

This project follows a complete machine learning lifecycle, as detailed in `Heart_Disease.ipynb`:

1.  **Data Loading:** Load the "Cleveland Heart Disease" dataset from `heart-disease-cleveland.csv`.
2.  **Exploratory Data Analysis (EDA):** Understand data distributions, find correlations, and visualize key patterns.
3.  **Data Preprocessing:**
      * Split data into training and testing sets (`train_test_split`).
      * Apply Feature Scaling using `StandardScaler` to ensure optimal model performance.
4.  **Model Training & Comparison:** Train four different models and evaluate them based on Accuracy and ROC AUC metrics.
5.  **Model Selection & Tuning:** `RandomForestClassifier` was selected as the best initial model and then optimized using `GridSearchCV` to find the best hyperparameters.
6.  **Model Saving:** Save the final model (`best_heart_model.joblib`) and the scaler (`scaler.joblib`) for use in the Streamlit application.

## 💻 Technologies Used

  * **Programming Language:** Python 3.9+
  * **Data Analysis & Manipulation:** Pandas, Numpy
  * **Data Visualization:** Matplotlib, Seaborn
  * **Automated EDA:** YData Profiling (Pandas Profiling)
  * **Machine Learning:** Scikit-learn
  * **Model Explainability (XAI):** SHAP
  * **Web Application:** Streamlit
  * **Model Serialization:** Joblib

## 📁 Project Structure

```
.
├── Heart_Disease.ipynb     # Notebook for data analysis and model training
├── streamlit_app.py        # Code for the Streamlit web application
├── best_heart_model.joblib # The trained model file (output from notebook)
├── scaler.joblib           # The scaler object file (output from notebook)
├── heart-disease-cleveland.csv # (Must be added) The dataset
├── requirements.txt        # (Recommended) File for required libraries
└── README.md               # This file
```

## ⚙️ Installation and Usage

To set up and run this project locally, follow these steps:

**1. Clone the Repository:**

```bash
git clone https://github.com/Ahmed-Al-Mohammadi/Heart_Disease.git
cd Heart_Disease
```

**2. (Optional) Create a Virtual Environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**3. Install Dependencies:**

(You should first create a `requirements.txt` file from the libraries used in `Heart_Disease.ipynb`)

```bash
pip install -r requirements.txt
```

*If you don't have a `requirements.txt` file, install the core libraries:*

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib ydata-profiling shap
```

**4. Run the Notebook (Optional):**

If you want to explore the data analysis and model training process yourself, run the Jupyter Notebook:

```bash
jupyter notebook Heart_Disease.ipynb
```

**Important:** You must run all cells in the notebook to generate the `best_heart_model.joblib` and `scaler.joblib` files, which are necessary for the app to run.

**5. Run the Streamlit App:**

To launch the interactive application:

```bash
streamlit run streamlit_app.py
```

This command will automatically open the application in your default web browser.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file (if it exists) for more details.


-----
