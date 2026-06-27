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
    """
    Train multinomial Logistic Regression on CNN feature vectors.

    Parameters
    ----------
    X_train : ndarray
        Feature matrix of shape (n_samples, 256)

    y_train : ndarray
        Integer class labels.

    Returns
    -------
    LogisticRegression
        Trained classifier.
    """
    
    classifier = LogisticRegression(
        solver="saga",              # Efficient solver for multinomial logistic regression
        multi_class="multinomial",  # Single softmax over all 5 classes
        max_iter=500,               # Raise to 500 if ConvergenceWarning appears
        C=1.0,                      # Inverse L2 regularisation strength
        class_weight="balanced",      # Adjust weights inversely proportional to class frequencies
        random_state=42,
        n_jobs=-1,                  # Parallelise across all CPU cores
        verbose=1,
    )
    if X_train.ndim != 2:
        raise ValueError(
            f"Expected a 2D feature matrix, got shape {X_train.shape} instead."
        )
    if len(X_train) != len(y_train):
        raise ValueError(
            "Number of feature vectors and labels must match"
        )
    classifier.fit(X_train, y_train)
    return classifier


def predict_lr(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    
    return model.predict(X)


def save_lr(model: LogisticRegression, filepath: str) -> None:
   
    joblib.dump(model, filepath)


def load_lr(filepath: str) -> LogisticRegression:
    
    return joblib.load(filepath)