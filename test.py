import mlflow
from mlflow.models import infer_signature
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
with mlflow.start_run():
    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = LogisticRegression()
    lr.fit(X, y)

    y_predict = lr.predict(X)
    signature = infer_signature(X, y_predict)
  
    params = lr.get_params()
    acc = lr.score(X, y)
    conf_matrix = confusion_matrix(y, y_predict, normalize='true')
    report = classification_report(y, y_predict)

    mlflow.log_params(params)

    true_positive = conf_matrix[0][0]
    true_negative = conf_matrix[1][1]
    false_positive = conf_matrix[0][1]
    false_negative = conf_matrix[1][0]

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("true_positive", true_positive)
    mlflow.log_metric("true_negative", true_negative)
    mlflow.log_metric("false_positive", false_positive)
    mlflow.log_metric("false_negative", false_negative)
    
    model_info = mlflow.sklearn.log_model(
        sk_model=lr, artifact_path="model", signature=signature
    )