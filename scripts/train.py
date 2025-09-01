import os
import pickle
import warnings

import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def train_and_save_model():
    """
    Loads data, trains an SVM model, and saves the trained model and scaler.
    It also prints the accuracy of the model on the training and testing sets.
    """
    print("---" * 12)
    print("Loading data...")
    try:
        # Load the diabetes dataset
        diabetes_data = pd.read_csv("../data/diabetes.csv")
    except FileNotFoundError:
        print(
            "Error: 'diabetes.csv' not found. Make sure the file is in the '../data/' directory."
        )
        return

    # Separate features and target
    X = diabetes_data.drop(columns=["Outcome"], axis=1)
    Y = diabetes_data["Outcome"]

    print("Training model...")

    # Data Standardization
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    X = standardized_data

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )

    # Train the SVM classifier
    classifier = svm.SVC(kernel="linear")
    classifier.fit(X_train, Y_train)

    # Evaluate the model on training data
    train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(train_prediction, Y_train)

    # Evaluate the model on test data
    test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(test_prediction, Y_test)

    print("Model training complete.")

    print("---" * 12)

    print(f"Training data Accuracy: {training_data_accuracy * 100:.2f}%")
    print(f"Test data Accuracy: {test_data_accuracy * 100:.2f}%")

    print("---" * 12)

    # Create the directory if it doesn't exist
    model_dir = "../model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the trained model and scaler
    try:
        with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
            pickle.dump({"classifier": classifier, "scaler": scaler}, f)
        print("Model saved successfully! :)")
    except IOError as e:
        print(f"Error: Unable to save the model. {e}")

    print("---" * 12)


if __name__ == "__main__":
    train_and_save_model()
