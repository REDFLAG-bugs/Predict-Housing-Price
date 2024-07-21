from sklearn.metrics import mean_absolute_error, mean_squared_error

def model_evalute(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    mae = mean_absolute_error(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    return mae, mse, Y_pred