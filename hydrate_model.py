# code assuming that proper csv files are imported 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

def load_and_combine_csvs(folder_path):
    """
    Load and combine all CSV files in the folder for training/testing.
    """
    print(f"Looking for files in: {os.path.abspath(folder_path)}")
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

    csv_files = [
        file for file in os.listdir(folder_path)
        if file.endswith('.csv') and not file.startswith('predictions')
    ]
    print(f"Files found: {csv_files}")
    
    if not csv_files:
        raise FileNotFoundError("No valid CSV files found in the specified folder.")
    
    data_frames = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        print(f"Loading {file_path}...")
        df = pd.read_csv(file_path)
        data_frames.append(df)
    
    combined_data = pd.concat(data_frames, ignore_index=True)
    print(f"Combined Data Shape: {combined_data.shape}")
    return combined_data

def preprocess_data(data):
    """
    Preprocess the data for training and testing.
    Includes handling missing values and feature engineering.
    """
    required_columns = [
        'Inj Gas Meter Volume Instantaneous',
        'Inj Gas Meter Volume Setpoint',
        'Inj Gas Valve Percent Open'
    ]
    for col in required_columns:
        if col not in data.columns:
            raise KeyError(f"Missing required column: {col}")
    
    print("\nHandling missing values...")
    data[required_columns] = data[required_columns].fillna(method="ffill").fillna(method="bfill")
    
    # Drop rows with remaining NaN values in required columns
    data = data.dropna(subset=required_columns).reset_index(drop=True)
    
    # Feature Engineering
    data['Volume_Difference'] = data['Inj Gas Meter Volume Instantaneous'] - data['Inj Gas Meter Volume Setpoint']
    
    # Define threshold dynamically (e.g., 95th percentile)
    threshold = data['Volume_Difference'].abs().quantile(0.95)
    data['Hydrate_Detected'] = (data['Volume_Difference'].abs() > threshold).astype(int)
    
    feature_cols = [
        'Volume_Difference',
        'Inj Gas Valve Percent Open',
        'Inj Gas Meter Volume Instantaneous',
        'Inj Gas Meter Volume Setpoint'
    ]
    X = data[feature_cols]
    y = data['Hydrate_Detected']
    
    print("\nClass Balance (Hydrate_Detected):")
    print(y.value_counts())
    return X, y, data

def train_model(X, y):
    """
    Train the Random Forest model with balanced class weights.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y)
    return model, scaler

def predict_hydrates(model, scaler, data):
    """
    Make predictions using the trained model and add them to the dataset.
    """
    data['Volume_Difference'] = data['Inj Gas Meter Volume Instantaneous'] - data['Inj Gas Meter Volume Setpoint']
    
    feature_cols = [
        'Volume_Difference',
        'Inj Gas Valve Percent Open',
        'Inj Gas Meter Volume Instantaneous',
        'Inj Gas Meter Volume Setpoint'
    ]
    X = data[feature_cols].fillna(method="ffill").fillna(method="bfill")
    X_scaled = scaler.transform(X)
    data['Hydrate_Predicted'] = model.predict(X_scaled)
    
    data['Prediction_Match'] = (data['Hydrate_Detected'] == data['Hydrate_Predicted']).astype(int)
    return data

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def evaluate_model(model, scaler, X_test, y_test):
    """
    Evaluate the model on test data and print a more detailed classification report.
    """
    # Scale the test set
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=["No Hydrate (0)", "Hydrate (1)"], zero_division=0)

    # Confusion matrix for detailed results
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print("\n********* CLASSIFICATION REPORT *********")
    print(report)
    print("*****************************************")


    print("\n********* CONFUSION MATRIX *********")
    print(f"                Predicted: No Hydrate (0) | Predicted: Hydrate (1)")
    print(f"Actual: No Hydrate (0) |       {cm[0, 0]:<5}               {cm[0, 1]:<5}")
    print(f"Actual: Hydrate (1)    |       {cm[1, 0]:<5}               {cm[1, 1]:<5}")
    print("*****************************************")

    print(f"\nOverall Test Set Accuracy: {accuracy:.2f}")


    print("\n********* CLASS-BY-CLASS METRICS *********")
    print(f"No Hydrate (0) -> Precision: {cm[0, 0] / (cm[0, 0] + cm[1, 0]):.2f}, Recall: {cm[0, 0] / (cm[0, 0] + cm[0, 1]):.2f}")
    print(f"Hydrate (1)    -> Precision: {cm[1, 1] / (cm[1, 1] + cm[0, 1]):.2f}, Recall: {cm[1, 1] / (cm[1, 1] + cm[1, 0]):.2f}")
    print("*****************************************")


def cross_validate_model(X, y):
    """
    Perform cross-validation to ensure the model generalizes well.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    print("\nPerforming 5-Fold Cross-Validation...")
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.2f}")

def visualize_predictions(data):
    """
    Visualize predictions using a scatterplot.
    """
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=data,
        x='Volume_Difference',
        y='Inj Gas Valve Percent Open',
        hue='Hydrate_Predicted', 
        palette={1: 'red', 0: 'blue'},
        alpha=0.6
    )
    plt.title("Hydrate Detection Scatterplot")
    plt.xlabel("Volume Difference")
    plt.ylabel("Inj Gas Valve Percent Open")
    plt.legend(title="Predicted Value")
    plt.grid(True)
    plt.show()

def main():

    folder_path = os.path.dirname(os.path.abspath(__file__))
    all_data = load_and_combine_csvs(folder_path)
    fearless_data = pd.read_csv(os.path.join(folder_path, "Fearless_709H-10_31-11_07.csv"))
    
    print("\nExcluding Fearless.csv for validation...")
    training_data = all_data[~all_data.index.isin(fearless_data.index)]
    
    print("\nPreprocessing training data...")
    X_train, y_train, processed_training_data = preprocess_data(training_data)
    
    print("\nTraining the model...")
    model, scaler = train_model(X_train, y_train)
    
    print("\nEvaluating on Fearless.csv...")
    X_fearless, y_fearless, processed_fearless = preprocess_data(fearless_data)
    evaluate_model(model, scaler, X_fearless, y_fearless)
    
    print("\nVisualizing predictions on Fearless.csv...")
    fearless_predictions = predict_hydrates(model, scaler, processed_fearless)
    visualize_predictions(fearless_predictions)

if __name__ == "__main__":
    main()
