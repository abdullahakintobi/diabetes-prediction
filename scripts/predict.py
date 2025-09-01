import os
import pickle
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def make_prediction():
    """
    Loads a pre-trained model and scaler to make a prediction based on user input.
    """
    # Load the trained model and scaler
    model_path = "../model/model.pkl"
    if not os.path.exists(model_path):
        print(
            "Error: 'model.pkl' not found. Please run train.py first to create the model."
        )
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    classifier = model["classifier"]
    scaler = model["scaler"]

    print("Please enter the following data for prediction:")

    try:
        # Collect user input for each feature
        pregnancies = float(input("Number of Pregnancies: "))
        glucose = float(input("Glucose Level: "))
        blood_pressure = float(input("Blood Pressure value: "))
        skin_thickness = float(input("Skin Thickness value: "))
        insulin = float(input("Insulin value: "))
        bmi = float(input("BMI value: "))
        diabetes_pedigree = float(input("Diabetes Pedigree Function value: "))
        age = float(input("Age of the person: "))

        # Create a list from the collected inputs
        input_data = [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age,
        ]
        print(f"\nInput data: {input_data}\n")

    except ValueError:
        print("\nError: Invalid input. Please enter numerical values for each field.")
        return

    # Change input data to a numpy array and reshape it
    input_data_array = np.asarray(input_data).reshape(1, -1)

    # Standardize input data
    input_data_std = scaler.transform(input_data_array)

    # Make prediction
    prediction = classifier.predict(input_data_std)

    print("---" * 10)

    if prediction[0] == 0:
        print(">> Diabetes: Negative (-ve)")
    else:
        print(">> Diabetes: Positive (+ve)")

    print("---" * 10)


if __name__ == "__main__":
    make_prediction()
