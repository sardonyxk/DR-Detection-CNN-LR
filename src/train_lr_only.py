import numpy as np
from classifier_lr import train_lr, predict_lr, save_lr
from evaluate import evaluate_model, plot_confusion_matrix

F_train_s = np.load("saved_model/F_train_s.npy")
F_val_s   = np.load("saved_model/F_val_s.npy")
F_test_s  = np.load("saved_model/F_test_s.npy")
y_train   = np.load("saved_model/y_train.npy")
y_val     = np.load("saved_model/y_val.npy")
y_test    = np.load("saved_model/y_test.npy")

lr_model = train_lr(F_train_s, y_train)
save_lr(lr_model, "saved_model/lr_model.joblib")

y_val_pred  = predict_lr(lr_model, F_val_s)
y_test_pred = predict_lr(lr_model, F_test_s)

val_metrics  = evaluate_model(y_val,  y_val_pred)
test_metrics = evaluate_model(y_test, y_test_pred)

plot_confusion_matrix(val_metrics["confusion_matrix"],
                      title="Validation Set",
                      filename="confusion_matrix_val.png")
plot_confusion_matrix(test_metrics["confusion_matrix"],
                      title="Test Set",
                      filename="confusion_matrix_test.png")

# training history curves — loads from json saved by train.py
import json
import matplotlib.pyplot as plt
from pathlib import Path

history_path = "saved_model/training_history.json"
if Path(history_path).exists():
    with open(history_path) as f:
        history = json.load(f)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"],     label="Train")
    plt.plot(history["val_accuracy"], label="Validation")
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"],     label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/training_curves.png", dpi=150)
    plt.close()
    print("  [OK] Training curves saved → results/training_curves.png")
else:
    print("  [WARN] training_history.json not found — skipping curves plot")