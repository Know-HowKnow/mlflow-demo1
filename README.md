# MLflow-demo1 
Demo MLflow Experiment with Sklearn model for sample ML modeling and tracking 

## Setup Environment
Using Anaconda Environment
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
## Remote Train by Python exe or Container Schedule
```
python train_model.py
```

## Experiment by Jupyter Notebook or Jupyter Lab
```
jupyter notebook
```