# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

def load_and_train_model(file_path='dataset3.xlsx'):
    # Load the dataset
    data = pd.read_excel(file_path, sheet_name='Sheet1')

    data = data.dropna(subset=['Column2', 'Column3'])  # Adjust based on critical columns

    # Handle missing values
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())

    categorical_cols = data.select_dtypes(include=['object']).columns
    data[categorical_cols] = data[categorical_cols].astype(str)
    data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
    data[categorical_cols] = data[categorical_cols].infer_objects()

    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Define features and target
    # target_column = ['Columnn', 'Columnn1', 'Columnn2', 'Column3']  # Replace with your actual target column
    X = data.iloc[:, :3]
    y = data.iloc[:, 5:9]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Train the model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Save the model and label encoders to disk for later use
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')

    return rf_model

def predict(input_data):
    # Load model and label encoders
    rf_model = joblib.load('random_forest_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')

    # Preprocess input_data as needed (assume it’s a list or array)
    input_data = [input_data]  # Adjust as needed based on your input
    prediction = rf_model.predict(input_data)

    return prediction[0]