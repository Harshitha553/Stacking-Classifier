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
# BASE LEARNERS
# -----------------------------------

lr = LogisticRegression(
    max_iter=1000
)

dt = DecisionTreeClassifier(
    max_depth=tree_depth,
    random_state=42
)

knn = KNeighborsClassifier(
    n_neighbors=knn_neighbors
)

# Train Individual Models

lr.fit(
    x_train,
    y_train
)

dt.fit(
    x_train,
    y_train
)

knn.fit(
    x_train,
    y_train
)

# Individual Predictions

lr_pred = lr.predict(
    x_test
)

dt_pred = dt.predict(
    x_test
)

knn_pred = knn.predict(
    x_test
)

# Individual Accuracies

lr_acc = accuracy_score(
    y_test,
    lr_pred
)

dt_acc = accuracy_score(
    y_test,
    dt_pred
)

knn_acc = accuracy_score(
    y_test,
    knn_pred
)

# -----------------------------------
# META LEARNER
# -----------------------------------

final_estimator = RandomForestClassifier(

    n_estimators=rf_estimators,

    max_depth=rf_depth,

    random_state=42
)
# -----------------------------------
# STACKING CLASSIFIER
# -----------------------------------

estimators = [

    ("lr", lr),

    ("dt", dt),

    ("knn", knn)

]

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
# STACKING PREDICTION
# -----------------------------------

stack_pred = model.predict(
    x_test
)

# -----------------------------------
# STACKING METRICS
# -----------------------------------

stack_acc = accuracy_score(
    y_test,
    stack_pred
)

stack_precision = precision_score(
    y_test,
    stack_pred
)

stack_recall = recall_score(
    y_test,
    stack_pred
)

stack_f1 = f1_score(
    y_test,
    stack_pred
)

st.subheader(
    "Stacking Classifier Performance"
)

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "Accuracy",
    round(stack_acc,3)
)

c2.metric(
    "Precision",
    round(stack_precision,3)
)

c3.metric(
    "Recall",
    round(stack_recall,3)
)

c4.metric(
    "F1 Score",
    round(stack_f1,3)
)
# -----------------------------------
# PERFORMANCE COMPARISON
# -----------------------------------

comparison = pd.DataFrame({

    "Model":[

        "Logistic Regression",

        "Decision Tree",

        "KNN",

        "Stacking Classifier"
    ],

    "Accuracy":[

        round(lr_acc,3),

        round(dt_acc,3),

        round(knn_acc,3),

        round(stack_acc,3)
    ]
})

st.subheader(
    "Performance Comparison"
)

st.dataframe(
    comparison
)

st.subheader(
    "Stacking Architecture"
)

st.info(
"""
Base Learners:
1. Logistic Regression
2. Decision Tree
3. KNN

Meta Learner:
4. Random Forest

The predictions of the base learners are used as input to the Random Forest meta learner to generate the final prediction.
"""
)
