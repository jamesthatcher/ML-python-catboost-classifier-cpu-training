# coding=utf-8
import json
import os
import sys
import catboost
from sklearn import datasets
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    log_loss,
    plot_confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Ensure the model is reproducible
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# get training data
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
class_names = breast_cancer.target_names
feat_names = breast_cancer.feature_names

# split train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, shuffle=True, random_state=RANDOM_SEED
)

# Define hyperparameters
params = {
    "depth": [1, 3, 7],
    "iterations": [100],
    "learning_rate": [0.01, 0.1, 0.2],
    "l2_leaf_reg": [1, 5, 10],
}

# Step 1: set up target metrics for evaluating training

# Define target loss metric to aim for
target_f1 = 0.7

# Instantiate classifier and run grid search to find best parameters
clf = catboost.CatBoostClassifier()
grid_clf = GridSearchCV(clf, params, scoring="neg_log_loss")
grid_clf.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_clf.best_estimator_
clf.fit(X_train, y_train)

# Step 3: Evaluate the quality of the trained model
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

# print classification report of classifier
print(classification_report(y_test, y_pred, target_names=class_names))

# evaluate the quality of the trained model using weighted f1 score
f1_metric = f1_score(y_test, y_pred, average="weighted")
print(f"f1 score: {round(f1_metric, 3)}")

# Only persist the model if we have passed our desired threshold
if f1_metric < target_f1:
    sys.exit("Training failed to meet threshold")

# Step 4: Persist the trained model in ONNX format in the local file system along with any significant metrics

# Persist model
clf.save_model(
    "model.onnx",
    format="onnx",
    export_parameters={
        "onnx_domain": "ai.catboost",
        "onnx_model_version": 1,
        "onnx_doc_string": "BinaryClassifier for breast_cancer dataset",
        "onnx_graph_name": "CatBoostModel_for_BinaryClassification",
    },
)

# calculate set of quality metrics
accuracy_metric = accuracy_score(y_test, y_pred)
logloss_metric = log_loss(y_test, y_pred_proba)
roc_auc_metric = roc_auc_score(y_test, y_pred)

# write metrics
if not os.path.exists("metrics"):
    os.mkdir("metrics")

with open("metrics/f1.metric", "w+") as f:
    json.dump(f1_metric, f)
with open("metrics/accuracy.metric", "w+") as f:
    json.dump(accuracy_metric, f)
with open("metrics/logloss.metric", "w+") as f:
    json.dump(logloss_metric, f)
with open("metrics/roc_auc.metric", "w+") as f:
    json.dump(roc_auc_metric, f)

# plots
confusion_metrics = plot_confusion_matrix(
    clf, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues
)
confusion_metrics.ax_.set_title(
    "Confusion matrix of CatBoostClassifier classifier on Breast Caner dataset"
)
plt.savefig("metrics/confusion_matrix.png")
plt.clf()
