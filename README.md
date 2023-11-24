# Setup Environment
## Anaconda Environment

```
conda create -n mlflow-demo python=3.8 --y
conda activate mlflow-demo
pip install -r requirements.txt
```

# MLflow Server
```
mlflow server --host 127.0.0.1 --port 8080
```

# ML Experiment Tracking
```
python train_model.py
```