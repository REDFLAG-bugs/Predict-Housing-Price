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
    data = data.apply(pd.to_numeric, errors='coerce')  #coverted all coloumns to numeric
    features = data.drop('MEDV', axis=1)
    target = data['MEDV']
    features_imputed = imputer.fit_transform(features)
    data_imputed = pd.DataFrame(features_imputed, columns=features.columns)
    data_imputed['MEDV'] = target.values
    return data_imputed

def outliners_remove(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    return data

def feature_engineering(data):
    data['RM^2'] = data['RM'] ** 2
    data['LSTAT^2'] = data['LSTAT'] ** 2
    return data


def feature_selection(data):
    correlation_matrix = data.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
    data = data.drop(to_drop, axis=1)
    return data


def scaler_features(data):
    scaler = StandardScaler()
    scale_data = scaler.fit_transform(data)
    return scale_data, scaler


#Splited the dataset into train and test data (80/20) 
def data_split(X,Y):
    return train_test_split(X,Y,test_size=0.2, random_state=42)