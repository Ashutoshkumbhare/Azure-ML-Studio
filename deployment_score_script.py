from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, auc
import pandas as pd
import joblib

from azureml.core.model import Model
from azureml.core import Workspace
from azureml.core import Run
import json
import os
import numpy as np

def init ():
    global model
    # all models in azure are registered in "AZUREML_MODEL_DIR" so the source path is given and the modelname
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'diabeticmodel1.pkl')
    model = joblib.load(model_path)

def run (raw_data):
    # will predect and return a list
    data = np.array(json. loads (raw_data)['data'])
    y_hat = model.predict(data)
    return y_hat.tolist()