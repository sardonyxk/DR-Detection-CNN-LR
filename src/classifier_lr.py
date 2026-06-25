"""
Logistic Regression classifier for Diabetic Retinopathy severity grading.

Classes:
0 = No DR
1 = Mild
2 = Moderate
3 = Severe
4 = Proliferative DR
"""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


def train_lr(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    classifier = LogisticRegression(
        solver="saga",              # Efficient stochastic solver for large feature spaces
        multi_class="multinomial",  # Single softmax over all 5 classes
        max_iter=200,               # Raise to 500 if ConvergenceWarning appears
        C=1.0,                      # Inverse L2 regularisation strength
        random_state=42,
        n_jobs=-1,                  # Parallelise across all CPU cores
        verbose=1,
    )

    classifier.fit(X_train, y_train)
    return classifier


def predict_lr(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    
    return model.predict(X)


def save_lr(model: LogisticRegression, filepath: str) -> None:
   
    joblib.dump(model, filepath)


def load_lr(filepath: str) -> LogisticRegression:
    
    return joblib.load(filepath)