# Iris Flower Classification 🌸

A simple machine learning web application built with Python and Streamlit to classify an Iris flower into its species.

## Project Description

This project uses the `Iris.csv` dataset and a **Logistic Regression** machine learning model.

The Streamlit application:

- Loads the Iris dataset.
- Removes the unnecessary `Id` column when it is present.
- Uses four flower measurements as input features.
- Splits the data into training and testing sets.
- Standardizes the features using `StandardScaler`.
- Trains a Logistic Regression model.
- Calculates model accuracy.
- Displays a confusion matrix.
- Displays a classification report.
- Allows the user to enter flower measurements and predict the Iris species.
- Displays prediction confidence.
- Shows the dataset in the application.

## Features

The model uses:

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

Target variable:

- Species

## Machine Learning Process

1. Load `Iris.csv`.
2. Remove the `Id` column if it exists.
3. Separate input features and target.
4. Split the data using an 80/20 stratified split.
5. Scale features using `StandardScaler`.
6. Create a Logistic Regression model.
7. Train the model.
8. Test the model.
9. Calculate accuracy and confusion matrix.
10. Predict the species entered by the user.

## Technologies Used

- Python
- NumPy
- Pandas
- Streamlit
- Scikit-learn
- Logistic Regression
- StandardScaler

## Project Structure

```text
iris-flower-classification/
│
├── app.py
├── Iris.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

Open a terminal in the project folder and run:

```bash
streamlit run app.py
```

## How to Use

1. Start the application.
2. Enter the Iris flower measurements.
3. Click **Predict Flower Species**.
4. View the predicted species and confidence.
5. View the model accuracy, confusion matrix, and classification report.

## Dataset

The application expects the dataset file to be named:

```text
Iris.csv
```

Expected feature columns:

```text
SepalLengthCm
SepalWidthCm
PetalLengthCm
PetalWidthCm
```

Target column:

```text
Species
```

## Model

**Algorithm:** Logistic Regression

**Feature Scaling:** StandardScaler

The model is trained directly when `app.py` runs, so a separate saved model file is not required.

## Author

Iris Flower Classification project.
