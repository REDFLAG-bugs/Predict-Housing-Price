# Housing Price Prediction Project

This project aims to predict housing prices using a machine learning model based on the [Boston Housing Dataset](https://www.kaggle.com/datasets/arunjangir245/boston-housing-dataset/data). The process includes data preprocessing, model training, evaluation, and visualization of results. The primary objective is to develop an accurate and interpretable model.

## Project Structure

```
Predict-Housing-Price/
├── Dataset/
│   └── Boston_Housing_Dataset.csv
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── model_evaluation.py
│   └── visualization.py
├── Report/
│   └── Housing_Price_Prediction_Report.pdf
├── main.py
├── Requirements.txt
└── README.md
```

### Folders and Files

- **Dataset/**: Contains the dataset used for training the model.
  - `housing.csv`: The Boston Housing Dataset.

- **src/**: Contains the source code for various stages of the project.
  - `data_preprocessing.py`: Script for data cleaning, handling missing values, removing outliers, feature engineering, and scaling.
  - `train.py`: Script for training the machine learning model.
  - `model_evaluation.py`: Script for evaluating the trained model.
  - `visualization.py`: Script for visualizing results such as feature importance, actual vs predicted prices, and residuals.

- **Report/**: Contains the detailed project report.
  - `Housing_Price_Prediction_Report.pdf`: The final report documenting the entire project, including methodology, results, and analysis.

- **main.py**: The main script to run the entire pipeline from data preprocessing to model training and evaluation.

- **Requirements.txt**: List of Python dependencies required to run the project.

## Getting Started

### Prerequisites

Ensure you have Python 3.6 or higher installed on your system. Install the required dependencies using:

```bash
pip install -r Requirements.txt
```

### Running the Project

1. **Data Preprocessing**: Clean and preprocess the data.
   ```bash
   python src/data_preprocessing.py
   ```

2. **Model Training**: Train the machine learning model.
   ```bash
   python src/train.py
   ```

3. **Model Evaluation**: Evaluate the trained model.
   ```bash
   python src/model_evaluation.py
   ```

4. **Visualization**: Generate visualizations for analysis.
   ```bash
   python src/visualization.py
   ```

Alternatively, you can run the entire pipeline using the `main.py` script:

```bash
python main.py
```

## Project Report

The detailed project report is available in the `Report/` folder. It includes comprehensive information about the data preprocessing steps, model training process, evaluation metrics, results, and analysis.

## Contributing

If you wish to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## References

- Dataset source: [Boston Housing Dataset](https://www.kaggle.com/datasets/arunjangir245/boston-housing-dataset/data)
- plotly : [Link](https://plotly.com/python/)
- Random Forest Regressor : [Link](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
