import os
from src.data_preprocessing import data_collection,handle_missing_values , outliners_remove, scaler_features, data_split
from src.train import train_model
from src.model_evalution import model_evalute
from src.visualization import plot_results, plot_feature_importance
from src.deployment import create_app, save_model, load_model


file_path = os.path.join('Datasets', 'Boston_Housing_Dataset.csv')
data= data_collection(file_path)
data = handle_missing_values(data)
data = outliners_remove(data)

X = data.drop('MEDV', axis=1)
Y= data['MEDV']

X_scaled, scaler = scaler_features(X)
X_train, X_test, Y_train, Y_test = data_split(X_scaled, Y)

best_model = train_model(X_train, Y_train)

mae, mse, Y_pred = model_evalute(best_model, X_test, Y_test)
print(f'Mean Absolute Err: {mae}')
print(f'Mean Squared Err: {mse}')

plot_results(Y_test, Y_pred)
plot_feature_importance(best_model, X.columns)

save_model(best_model, scaler)