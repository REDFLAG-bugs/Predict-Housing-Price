import matplotlib.pyplot as plt
import numpy as np


def plot_results(Y_test, Y_pred):
    plt.figure(figsize=(10,6))
    plt.scatter(Y_test, Y_pred, alpha=0.7)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    plt.show()


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10,6))
    plt.title('Feature Importance')
    plt.bar(range(len(feature_names)), importances[indices], align='center')
    plt.xticks(range(len(feature_names)), feature_names, rotation = 90)
    plt.tight_layout()
    plt.show()
