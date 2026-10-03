# Fraud Detection Using Machine Learning

A machine learning project for detecting fraudulent financial transactions using transaction-level behavioral and balance information. The project explores transaction patterns, addresses class imbalance using SMOTE, compares multiple classification algorithms, and applies hyperparameter optimization to improve fraud detection performance.

## Project Overview

Fraud detection is a critical problem in banking and fintech because fraudulent transactions can result in significant financial losses.

This project builds a classification-based fraud detection system that predicts whether a financial transaction is fraudulent or legitimate.

The workflow includes:

- Exploratory Data Analysis
- Data preprocessing
- Categorical feature encoding
- Feature scaling
- Class imbalance handling using SMOTE
- Train-test splitting
- Logistic Regression
- Random Forest
- Gradient Boosting
- GridSearchCV
- RandomizedSearchCV
- Model performance comparison
- Confusion matrix analysis

## Problem Statement

The objective is to identify potentially fraudulent financial transactions based on transaction characteristics such as transaction type, transaction amount, account balances, and sender/receiver information.

The target variable is `isFraud`:

- `0` = Legitimate transaction
- `1` = Fraudulent transaction

## Dataset

The dataset contains **11,142 transactions and 10 columns**.

### Features

| Feature | Description |
|---|---|
| `step` | Unit of simulated time, where one step represents one hour |
| `type` | Transaction type |
| `amount` | Transaction amount |
| `nameOrig` | Customer who initiated the transaction |
| `oldbalanceOrg` | Sender's balance before the transaction |
| `newbalanceOrig` | Sender's balance after the transaction |
| `nameDest` | Recipient of the transaction |
| `oldbalanceDest` | Recipient's balance before the transaction |
| `newbalanceDest` | Recipient's balance after the transaction |
| `isFraud` | Target variable indicating fraudulent transaction |

### Transaction Types

The dataset contains transaction types including:

- CASH-IN
- CASH-OUT
- DEBIT
- PAYMENT
- TRANSFER

## Machine Learning Workflow

```text
Raw Transaction Data
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Data Preprocessing
        │
        ├── Categorical Encoding
        └── Feature Scaling
        │
        ▼
Class Imbalance Handling
        │
        └── SMOTE
        │
        ▼
Train-Test Split
        │
        ▼
Model Training
        │
        ├── Logistic Regression
        ├── Random Forest
        └── Gradient Boosting
        │
        ▼
Hyperparameter Optimization
        │
        ├── GridSearchCV
        └── RandomizedSearchCV
        │
        ▼
Model Evaluation
        │
        ├── Accuracy
        ├── Precision
        ├── Recall
        ├── F1 Score
        └── Confusion Matrix
```

## Data Preprocessing

The preprocessing pipeline includes:

### 1. Categorical Encoding

Categorical transaction information is converted into numerical form using `LabelEncoder`.

### 2. Feature Scaling

`StandardScaler` is used to standardize numerical features.

### 3. Handling Class Imbalance

Fraud detection datasets can contain an imbalance between fraudulent and legitimate transactions.

SMOTE (Synthetic Minority Oversampling Technique) is used to generate synthetic samples for the minority class.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='auto', random_state=42)

X_res, y_res = smote.fit_resample(X, y)
```

## Models Used

### Logistic Regression

Logistic Regression is used as the baseline classification model.

It provides an interpretable baseline for determining whether transaction characteristics can distinguish fraudulent transactions.

### Random Forest

Random Forest is used as an ensemble classification model capable of learning nonlinear relationships between transaction features.

Two hyperparameter optimization approaches were evaluated:

- GridSearchCV
- RandomizedSearchCV

### Gradient Boosting

Gradient Boosting is evaluated as another ensemble learning approach that builds models sequentially to improve classification performance.

Both GridSearchCV and RandomizedSearchCV were used for optimization.

## Hyperparameter Optimization

### Random Forest

The project searches parameters including:

```python
{
    'n_estimators': [50, 100, 200, 300],
    'bootstrap': [True, False],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
```

RandomizedSearchCV evaluates a randomized subset of parameter combinations.

### Gradient Boosting

The project explores:

```python
{
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'n_estimators': [100, 200, 300]
}
```

## Model Performance

The notebook compares the following models:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 96.35% | 99.56% | 92.96% | 96.15% |
| Random Forest + GridSearchCV | 99.93% | 100.00% | 99.85% | 99.92% |
| Random Forest + RandomizedSearchCV | **99.95%** | **100.00%** | **99.90%** | **99.95%** |
| Gradient Boosting + GridSearchCV | 99.90% | 99.95% | 99.85% | 99.90% |
| Gradient Boosting + RandomizedSearchCV | 99.93% | 100.00% | 99.85% | 99.92% |

According to the notebook's evaluation, Random Forest with RandomizedSearchCV produced the highest F1 score.

## Evaluation Metrics

### Accuracy

Measures the proportion of correctly classified transactions.

### Precision

Measures how many transactions predicted as fraudulent were actually fraudulent.

### Recall

Measures how many actual fraudulent transactions were successfully detected.

Recall is particularly important in fraud detection because missing fraudulent transactions can result in financial losses.

### F1 Score

The F1 score combines precision and recall into a single metric and is useful when evaluating classification performance across imbalanced classes.

## Confusion Matrix

The project generates confusion matrices for the tuned Random Forest and Gradient Boosting models to visualize:

- True Positives
- True Negatives
- False Positives
- False Negatives

These results help analyze how effectively the models distinguish fraudulent and legitimate transactions.

## Technologies Used

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- Logistic Regression
- Random Forest
- Gradient Boosting
- GridSearchCV
- RandomizedSearchCV

### Imbalanced Learning

- Imbalanced-learn
- SMOTE

## Project Structure

```text
Fraud-Detection/
│
├── Fraud Detection.ipynb
├── Fraud_Analysis_Dataset.csv
├── README.md
└── requirements.txt
```

A more production-oriented structure can later be created:

```text
Fraud-Detection/
│
├── data/
│   └── Fraud_Analysis_Dataset.csv
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── fraud_detection_model.pkl
│
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Fraud Detection.ipynb
```

## Running on Google Colab

The notebook can also be executed using Google Colab.

1. Open Google Colab.
2. Upload `Fraud Detection.ipynb`.
3. Upload the dataset.
4. Update the dataset path.
5. Run the notebook cells sequentially.

## Key Findings

The notebook found that:

- Logistic Regression provides a useful baseline but has lower recall than the ensemble models.
- Random Forest and Gradient Boosting substantially improve fraud classification performance.
- Hyperparameter optimization improves the evaluated model performance.
- Random Forest with RandomizedSearchCV achieved the highest reported F1 score in the notebook.
- Recall is an important metric for this problem because undetected fraudulent transactions represent potential financial losses.

## Important Methodological Note

The current notebook applies SMOTE before the train-test split.

For a more rigorous machine learning evaluation, the recommended approach is:

```text
Original Dataset
      │
      ▼
Train-Test Split
      │
      ├── Training Data
      │       │
      │       └── SMOTE
      │
      └── Test Data
              │
              └── Keep Original Distribution
```

This prevents synthetic samples derived from the training data from influencing the test set and provides a more reliable estimate of model performance.

## Future Improvements

Potential improvements include:

- Apply SMOTE only to the training data.
- Use a preprocessing pipeline with `Pipeline` or `imblearn.Pipeline`.
- Evaluate using ROC-AUC and PR-AUC.
- Perform stratified cross-validation.
- Analyze feature importance.
- Add SHAP-based model explainability.
- Build a real-time fraud prediction API.
- Create a Streamlit dashboard.
- Save the trained model using Joblib.
- Add transaction-level prediction functionality.
- Add cost-sensitive evaluation based on false-positive and false-negative costs.
- Deploy the model as a web application or API.

## Disclaimer

This project is an educational machine learning implementation using a simulated financial transaction dataset. It should not be considered a production-ready fraud detection system for real financial institutions.

## Author

**Jamshed Ahmad**

Data Science & Machine Learning

---

⭐ If you find this project useful, consider giving the repository a star.
