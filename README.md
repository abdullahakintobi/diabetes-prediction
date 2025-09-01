# Diabetes Prediction using Support Vector Machine

![Python](https://img.shields.io/badge/Python-v3.11-blue?logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.7.1-red?logo=scikit-learn&logoColor=white) ![Status](https://img.shields.io/badge/Status-Active-brightgreen) ![Model](https://img.shields.io/badge/Model-SVM-orange)

This project uses a **Support Vector Machine (SVM)** model to predict the likelihood of diabetes based on health indicators such as glucose level, BMI, age, blood pressure, and more.


## Workflow

The project follows a structured machine learning workflow:

![Workflow Diagram](image/workflow.png)



## Project Structure

```
diabetes-prediction/
├── data/                 # Dataset (diabetes.csv)
├── image/                # Project diagrams and visualizations
├── model/                # Trained ML model
├── notebook/             # Jupyter notebooks for EDA & prototyping
├── report/               # Reports and analysis results
├── scr/                  # Python scripts (train & predict)
├── pyproject.toml        # Project dependencies (managed with uv)
├── uv.lock               # Dependency lock file
└── README.md             # Project documentation
```



## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/abdullahakintobi/diabetes-prediction.git
cd diabetes-prediction
```

### 2. Create Virtual Environment with uv

```bash
uv venv
```

### 3. Install Dependencies

```bash
uv pip install -r pyproject.toml
```



## Usage

### 1. Training the Model

The `train.py` script loads the data, trains an SVM classifier, evaluates its performance, and saves the trained model and scaler to the `model/` directory.

To train the model, run the following command from the project's root directory:

```bash
python scr/train.py
```

### Output:

```bash
Loading data...
Training model...
Model training complete.
Training data Accuracy: 78.66%
Test data Accuracy: 77.27%
Model saved successfully!
```



### 2. Making Predictions

The `predict.py` script loads the saved model and prompts the user to enter new data. It then standardizes the input and uses the model to predict the outcome.

To make a prediction, run the script and follow the on-screen prompts:

```bash
python scr/predict.py
```

### Example interaction:

```bash
Please enter the following data for prediction:
Number of Pregnancies: 5
Glucose Level: 189
Blood Pressure value: 64
Skin Thickness value: 33
Insulin value: 325
BMI value: 31.2
Diabetes Pedigree Function value: 0.583
Age of the person: 29

Input data: [5.0, 189.0, 64.0, 33.0, 325.0, 31.2, 0.583, 29.0]

Diabetes: Positive (+ve)
```



## Dataset

The dataset used is the **[Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)**, stored in `data/diabetes.csv`.

* **Features**: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
* **Target**: Outcome (0 = Negative, 1 = Positive)



## Model Performance

* Classifier: **SVM (Support Vector Machine, Linear Kernel)**
* Example run results:

  * Training Accuracy: \~78–80%
  * Test Accuracy: \~75–77%



## Tech Stack

* Python 3.11+
* uv (for environment management)
* scikit-learn
* pandas, numpy
* Jupyter Notebook (for EDA and experiments)



## Future Improvements

* Add GUI or Web App (Streamlit/Gradio).
* Hyperparameter tuning for SVM (GridSearchCV).
* Experiment with other models (Random Forest, XGBoost, Neural Networks).
* Deploy the model with FastAPI/Flask + Docker.



## Contributing

Contributions, issues, and feature requests are welcome.

* Fork the repo
* Create a new branch (`feature-branch`)
* Commit your changes
* Open a Pull Request



## License

This project is licensed under the terms of the [Apache-2.0 license](./LICENSE).


### Author: [Abdullah Akintobi](https://www.linkedin.com/in/abdullahakintobi/)