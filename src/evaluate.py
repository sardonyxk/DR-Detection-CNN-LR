import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute accuracy, classification report, and confusion matrix."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true has {len(y_true)} samples but y_pred has {len(y_pred)}."
        )
    accuracy = accuracy_score(y_true, y_pred)

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,          # Four decimal places for academic reporting
        zero_division=0,   # Suppress warnings for unseen classes in small splits
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    
    separator = "=" * 65

    print(f"\n{separator}")
    print("  EVALUATION RESULTS")
    print(separator)

    # 1. Accuracy
    print(f"\n  Overall Accuracy : {accuracy * 100:.2f}%\n")

    # 2. Classification report
    print("  Classification Report")
    print("  " + "-" * 63)
    
    for line in str(report_str).splitlines():
        print(f"  {line}")

    # 3. Confusion matrix — plain text grid, rows = true, columns = predicted
    print(f"\n  Confusion Matrix")
    print("  " + "-" * 63)
    print("  Rows = True label   |   Columns = Predicted label\n")

    
    col_labels = "".join(f"  [{i}]" for i in range(len(CLASS_NAMES)))
    print(f"  {'':>18}{col_labels}")    

    
    for true_idx, row in enumerate(cm):
        label   = CLASS_NAMES[true_idx]
        values  = "".join(f"{v:5d} " for v in row)
        print(f"  True [{true_idx}] {label:<12}|  {values}")

    print(f"\n{separator}\n")

    return {
        "accuracy":              accuracy,
        "classification_report": report_str,
        "confusion_matrix":      cm,
    }
    
def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    output_dir: str = "results",
    filename: str = "confusion_matrix.png"
)-> None:
    """Plot the confusion as a heatmap and save it to a file."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(output_dir) / filename
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        linecolor="gray",
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")
    