import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def data_collection(file_path):
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data= pd.read_csv(file_path, header=None, names=column_names)
    return data

def handle_missing_values(data):
    imputer = SimpleImputer(strategy='mean')
    data = data.apply(pd.to_numeric, errors='coerce')
    features = data.drop('MEDV', axis=1)
    target = data['MEDV']
    features_imputed = imputer.fit_transform(features)
    data_imputed = pd.DataFrame(features_imputed, columns=features.columns)
    data_imputed['MEDV'] = target.values
    return data_imputed

def outliners_remove(data):
    return data[(np.abs(data - data.mean()) <= (3 * data.std())).all(axis=1)]

def scaler_features(data):
    scaler = StandardScaler()
    scale_data = scaler.fit_transform(data)
    return scale_data, scaler

def data_split(X,Y):
    return train_test_split(X,Y,test_size=0.2, random_state=42)