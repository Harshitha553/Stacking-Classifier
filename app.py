import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Stacking Classifier",
    layout="wide"
)

st.title(
    "🩺 Disease Prediction using Stacking Classifier"
)

# -----------------------------------
# CREATE FOLDERS
# -----------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "data",
    exist_ok=True
)

# -----------------------------------
# LOAD DATASET
# -----------------------------------

data = load_breast_cancer()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["Target"] = data.target

dataset_path = "data/breast_cancer.csv"

if not os.path.exists(dataset_path):

    df.to_csv(
        dataset_path,
        index=False
    )

# -----------------------------------
# DISPLAY DATA
# -----------------------------------

st.subheader("Dataset")

st.dataframe(
    df.head()
)

st.write(
    "Shape:",
    df.shape
)

st.write(
    "Missing Values:"
)

st.write(
    df.isnull().sum()
)

st.write(
    "Statistics"
)

st.write(
    df.describe()
)

# -----------------------------------
# FEATURES & TARGET
# -----------------------------------

x = df.drop(
    "Target",
    axis=1
)

y = df["Target"]

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

x_train,x_test,y_train,y_test = (
    train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )
)

# -----------------------------------
# SCALING
# -----------------------------------

scaler = StandardScaler()

x_train = scaler.fit_transform(
    x_train
)

x_test = scaler.transform(
    x_test
)

pickle.dump(
    scaler,
    open(
        "models/scaler.pkl",
        "wb"
    )
)

# -----------------------------------
# HYPERPARAMETERS
# -----------------------------------

st.sidebar.header(
    "Stacking Hyperparameters"
)

knn_neighbors = st.sidebar.slider(
    "KNN Neighbors",
    1,
    20,
    5
)

tree_depth = st.sidebar.slider(
    "Decision Tree Depth",
    1,
    20,
    5
)

rf_estimators = st.sidebar.slider(
    "Random Forest Trees",
    10,
    300,
    100
)

rf_depth = st.sidebar.slider(
    "Random Forest Depth",
    1,
    20,
    10
)

# -----------------------------------
# BASE MODELS
# -----------------------------------

estimators = [

    (
        "lr",
        LogisticRegression(
            max_iter=1000
        )
    ),

    (
        "dt",
        DecisionTreeClassifier(
            max_depth=tree_depth,
            random_state=42
        )
    ),

    (
        "knn",
        KNeighborsClassifier(
            n_neighbors=knn_neighbors
        )
    )

]

# -----------------------------------
# META MODEL
# -----------------------------------

final_estimator = RandomForestClassifier(

    n_estimators=rf_estimators,
    max_depth=rf_depth,
    random_state=42
)

# -----------------------------------
# STACKING MODEL
# -----------------------------------

model = StackingClassifier(

    estimators=estimators,

    final_estimator=final_estimator,

    cv=5
)

model.fit(
    x_train,
    y_train
)

pickle.dump(
    model,
    open(
        "models/stacking_model.pkl",
        "wb"
    )
)

# -----------------------------------
# PREDICTIONS
# -----------------------------------

y_pred = model.predict(
    x_test
)

# -----------------------------------
# METRICS
# -----------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

st.subheader(
    "Model Performance"
)

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "Accuracy",
    round(accuracy,3)
)

c2.metric(
    "Precision",
    round(precision,3)
)

c3.metric(
    "Recall",
    round(recall,3)
)

c4.metric(
    "F1 Score",
    round(f1,3)
)

# -----------------------------------
# USER INPUT
# -----------------------------------

st.subheader(
    "Predict Disease"
)

mean_radius = st.slider(
    "Mean Radius",
    5.0,
    30.0,
    15.0
)

mean_texture = st.slider(
    "Mean Texture",
    5.0,
    40.0,
    20.0
)

mean_perimeter = st.slider(
    "Mean Perimeter",
    40.0,
    200.0,
    100.0
)

mean_area = st.slider(
    "Mean Area",
    100.0,
    2500.0,
    1000.0
)

input_data = pd.DataFrame({

    "mean radius":[mean_radius],
    "mean texture":[mean_texture],
    "mean perimeter":[mean_perimeter],
    "mean area":[mean_area]

})

for col in x.columns:

    if col not in input_data.columns:

        input_data[col] = df[col].mean()

input_scaled = scaler.transform(
    input_data
)

if st.button(
    "Predict"
):

    prediction = model.predict(
        input_scaled
    )

    if prediction[0] == 1:

        st.success(
            "Disease Detected"
        )

    else:

        st.success(
            "No Disease Detected"
        )