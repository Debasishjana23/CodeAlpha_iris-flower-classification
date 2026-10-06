import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("Iris.csv")


# ==========================================
# 2. REMOVE UNNECESSARY ID COLUMN
# ==========================================

if "Id" in data.columns:
    data = data.drop("Id", axis=1)


# ==========================================
# 3. SEPARATE INPUT AND TARGET
# ==========================================

X = data[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
]

y = data["Species"]


# ==========================================
# 4. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# ==========================================
# 6. CREATE MODEL
# ==========================================

model = LogisticRegression(
    max_iter=1000
)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

model.fit(
    X_train,
    y_train
)


# ==========================================
# 8. TEST MODEL
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ==========================================
# 9. STREAMLIT PAGE
# ==========================================

st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸"
)


st.title("🌸 Iris Flower Classification")

st.write(
    "Enter the measurements of an Iris flower "
    "to predict its species."
)


# ==========================================
# 10. SIDEBAR
# ==========================================

st.sidebar.title("Model Information")

st.sidebar.write(
    "Dataset: Iris.csv"
)

st.sidebar.write(
    "Model: Logistic Regression"
)

st.sidebar.write(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ==========================================
# 11. USER INPUT
# ==========================================

st.subheader("Enter Flower Measurements")


sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.1,
    step=0.1
)


sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5,
    step=0.1
)


petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4,
    step=0.1
)


petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2,
    step=0.1
)


# ==========================================
# 12. PREDICTION
# ==========================================

if st.button("🔍 Predict Flower Species"):

    new_flower = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])


    # Scale input
    new_flower_scaled = scaler.transform(
        new_flower
    )


    # Prediction
    prediction = model.predict(
        new_flower_scaled
    )


    # Probability
    probabilities = model.predict_proba(
        new_flower_scaled
    )


    predicted_species = prediction[0]

    confidence = (
        max(probabilities[0]) * 100
    )


    # ======================================
    # DISPLAY RESULT
    # ======================================

    st.success(
        f"🌸 Predicted Species: {predicted_species}"
    )

    st.info(
        f"Prediction Confidence: {confidence:.2f}%"
    )


# ==========================================
# 13. MODEL ACCURACY
# ==========================================

st.subheader("📊 Model Performance")

st.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)


# ==========================================
# 14. CONFUSION MATRIX
# ==========================================

st.subheader("Confusion Matrix")

species_names = sorted(
    y.unique()
)

cm_df = pd.DataFrame(
    cm,
    index=species_names,
    columns=species_names
)

st.dataframe(cm_df)


# ==========================================
# 15. CLASSIFICATION REPORT
# ==========================================

st.subheader("📈 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df
)


# ==========================================
# 16. SHOW DATASET
# ==========================================

st.subheader("📋 Dataset")

st.dataframe(data)