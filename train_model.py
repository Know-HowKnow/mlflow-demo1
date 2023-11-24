import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def load_dataset(test_size=0.2, random_state=42):
    # Load the Iris dataset
    x, y = datasets.load_iris(return_X_y=True)
    
    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state)

    return x_train, x_test, y_train, y_test

def train_model(x_train, y_train, 
                params={}):
    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(x_train, y_train)

    return lr

def evaluate_model(model, x_test, y_test):
    # Predict on the test set
    y_pred = model.predict(x_test)

    # Calculate accuracy as a target loss metric
    conf_matrix = confusion_matrix(y_test, y_pred, normalize='true')
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return conf_matrix, report

# Model Training
hyper_param = {"max_iter": 1000, "random_state": 42}

x_train, x_test, y_train, y_test = load_dataset()
model = train_model(x_train, y_train, params=hyper_param)
conf_matrix, class_report = evaluate_model(model, x_test, y_test)

print(conf_matrix)
print(class_report)

metrics = class_report["weighted avg"]
metrics["accuracy"] = class_report["accuracy"]

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("MLflow Sklearn")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(hyper_param)

    # Log the loss metric
    mlflow.log_metrics(metrics)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # Infer the model signature
    signature = infer_signature(x_train, model.predict(x_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="iris_model",
        signature=signature,
        input_example=x_test,
        registered_model_name="LR-iris1",
    )