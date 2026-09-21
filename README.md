# Diabetes Prediction

A machine learning project for predicting diabetes outcomes using patient health-related features.

The project uses the **Pima Indians Diabetes Dataset** and compares two classification algorithms:

* Random Forest
* Logistic Regression

The goal of this project is to practice data analysis, model training, evaluation, and feature importance using Python and Scikit-learn.

---

## Features

* Load and inspect a real-world diabetes dataset
* Check dataset shape and missing values
* Separate features and target variable
* Split data into training and testing sets
* Train a Random Forest classifier
* Train a Logistic Regression classifier
* Compare model accuracy
* Generate classification reports
* Generate confusion matrices
* Analyze Random Forest feature importance
* Visualize model results
* Display sample predictions

---

## Dataset

The project uses the **Pima Indians Diabetes Dataset**.

The dataset contains **768 records** and **8 input features**.

### Features

| Feature                  | Description                  |
| ------------------------ | ---------------------------- |
| Pregnancies              | Number of pregnancies        |
| Glucose                  | Plasma glucose concentration |
| BloodPressure            | Diastolic blood pressure     |
| SkinThickness            | Triceps skin fold thickness  |
| Insulin                  | 2-Hour serum insulin         |
| BMI                      | Body mass index              |
| DiabetesPedigreeFunction | Diabetes pedigree function   |
| Age                      | Age of the patient           |

### Target

`Outcome`

* `0` → No Diabetes
* `1` → Diabetes

---

## Machine Learning Models

### Random Forest

The project uses a Random Forest classifier with:

* `200` trees
* `min_samples_leaf=2`
* `random_state=42`

### Logistic Regression

Logistic Regression is used as a second model to provide a comparison with Random Forest.

---

## Results

The dataset is divided into:

* **80% training data**
* **20% testing data**

The test set contains 154 samples.

### Accuracy

| Model               | Accuracy |
| ------------------- | -------: |
| Random Forest       |   74.68% |
| Logistic Regression |   71.43% |

The project also generates classification reports and confusion matrices for both models.

---

## Feature Importance

Random Forest feature importance shows the relative contribution of each input feature to the model.

The most important features in this experiment were:

1. Glucose
2. BMI
3. Age
4. DiabetesPedigreeFunction

The exact importance values can be viewed when running the project.

---

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

## Project Structure

```text
diabet/
│
├── diabetes.csv
├── diabet.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository:

```command-line
git clone https://github.com/AmirPR0/diabetes-prediction.git
```

Move into the project directory:

```command-line
cd diabetes-prediction
```

Install the required dependencies:

```command-line
pip install -r requirements.txt
```

---

## Run the Project

Run the Python script:

```command-line
python diabet.py
```

The program displays:

* Dataset information
* Model accuracy
* Classification reports
* Confusion matrices
* Feature importance values
* Sample predictions

It also displays visualizations for feature importance and the confusion matrix.

---

## Limitations

This project is intended for educational purposes.

The model is trained on a relatively small dataset and should not be considered a medical diagnostic system. The predictions are not intended for clinical decision-making.

---

## Author

**Amirhossien Rahimi moghaddam**

GitHub: [AmirPR0](https://github.com/AmirPR0)
