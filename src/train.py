import os
import sys
import time
import argparse
import warnings
from pathlib import Path

from matplotlib.pyplot import bar
import numpy as np
import cv2
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info/warning logs
warnings.filterwarnings("ignore")

import tensorflow as tf

from model_cnn     import build_cnn_feature_extractor  # CNN architecture only
from classifier_lr import train_lr, predict_lr, save_lr # LR train/predict/save
from evaluate    import evaluate_model                 # metrics & printing

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

TARGET_SIZE = (224, 224)       # (H, W) — matches CNN Input(shape=(224,224,3))
INPUT_SHAPE = (224, 224, 3)    # passed to build_cnn_feature_extractor()
SUPPORTED   = {".jpg", ".jpeg", ".png", ".bmp"}

EXPECTED_FEATURE_DIM = 256 

CLASS_MAP = {               
    "No_DR"           : 0,
    "Mild"            : 1,
    "Moderate"        : 2,
    "Severe"          : 3,
    "Proliferative_DR": 4,
}

CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

# STEP 1 — Data Loading


def load_image(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        print(f"  [WARN] Cannot read: {path.name}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Safety resize — should be a no-op if preprocessing.py ran correctly
    if img.shape[:2] != TARGET_SIZE:
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

    return img.astype(np.float32) / 255.0

def load_dataset(data_dir: str) -> tuple[list[Path], np.ndarray]:
    root = Path(data_dir)

    if not root.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {root.resolve()}\n"
            "Run preprocessing.py first to create it."
        )

    image_paths = []
    labels = []

    print(f"\n{'='*65}")
    print(f"  STEP 1 — Loading dataset from: {root.resolve()}")
    print(f"{'='*65}")

    for class_name, label in CLASS_MAP.items():
        class_dir = root / class_name

        if not class_dir.is_dir():
            print(f"  [WARN] Missing class folder: {class_name}")
            continue

        files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED
        ]

        print(f"  {class_name:<22}  {len(files):>5} file(s)")

        for fpath in files:
            image_paths.append(fpath)
            labels.append(label)

    labels = np.array(labels, dtype=np.int32)

    print(f"\n  Total images : {len(image_paths):>6}")
    print(f"  Labels       : {len(labels):>6}")

    return image_paths, labels


# STEP 2 — Train / Validation / Test Split

from sklearn.model_selection import train_test_split

def split_dataset(
    image_paths: list,
    labels: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
):
    print(f"\n{'='*65}")
    print(f"  STEP 2 — Splitting dataset (PATH-BASED)")
    print(f"{'='*65}")

    # Step 1: train + temp split
    train_paths, temp_paths, y_train, y_temp = train_test_split(
        image_paths,
        labels,
        test_size=(val_size + test_size),
        stratify=labels,
        random_state=42
    )

    # Step 2: split temp → val + test
    val_ratio = val_size / (val_size + test_size)

    val_paths, test_paths, y_val, y_test = train_test_split(
        temp_paths,
        y_temp,
        test_size=(1 - val_ratio),
        stratify=y_temp,
        random_state=42
    )

    print(f"  Train      : {len(train_paths):>6}")
    print(f"  Validation : {len(val_paths):>6}")
    print(f"  Test       : {len(test_paths):>6}")

    return train_paths, val_paths, test_paths, y_train, y_val, y_test

#Step 3: CNN training
def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    epochs: int = 20,
) -> tf.keras.Model:

    print(f"\n{'='*65}")
    print("  STEP 3 — CNN Training (Adam Optimizer)")
    print(f"{'='*65}")

    #Build CNN feature extractor model
    cnn = build_cnn_feature_extractor(input_shape=INPUT_SHAPE)
    
    #Temporay classification head
    model=tf.keras.Sequential([
        cnn,
        tf.keras.layers.Dense(5, activation='softmax', name="classifier")
    ])
    
    #compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    #Train
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )
    print("\n  CNN training complete.")
    return cnn

# STEP 4 — CNN Feature Extraction


def _extract_split(
    cnn: tf.keras.Model,
    image_paths: list,
    labels: np.ndarray,
    batch_size: int,
    split_name: str = "",
) -> tuple[np.ndarray, np.ndarray]:

    n = len(image_paths)
    features = []
    y_batch = []

    n_batches = (n + batch_size - 1) // batch_size
    tag = f"[{split_name}] " if split_name else ""

    print(f"  {tag}Extracting {n} images in {n_batches} batches ...")

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n)

        batch_paths = image_paths[start:end]
        batch_labels = labels[start:end]

        batch_images = []

        for path in batch_paths:
            img = load_image(path)
            batch_images.append(img)

        batch_images = np.array(batch_images, dtype=np.float32)

        feats = cnn.predict(batch_images, verbose=0)

        features.append(feats)
        y_batch.append(batch_labels)

        print(f"{tag}Processed {end}/{n} ({end/n*100:.1f}%)")

    print()

    return np.vstack(features), np.concatenate(y_batch)


def extract_all_features(
    cnn: tf.keras.Model,
    train_paths: list,
    val_paths: list,
    test_paths: list,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
):
    print(f"\n{'='*65}")
    print(f"  STEP 3 — CNN Feature Extraction (STREAMING)")
    print(f"{'='*65}")

    F_train, y_train = _extract_split(cnn, train_paths, y_train, batch_size, "Train")
    F_val, y_val     = _extract_split(cnn, val_paths,   y_val,   batch_size, "Val")
    F_test, y_test   = _extract_split(cnn, test_paths,  y_test,  batch_size, "Test")

    print(f"\n  Feature shapes:")
    print(f"  Train: {F_train.shape}")
    print(f"  Val  : {F_val.shape}")
    print(f"  Test : {F_test.shape}")

    return F_train, F_val, F_test, y_train, y_val, y_test


# STEP 4 — Feature Scaling

def scale_features(
    F_train: np.ndarray,
    F_val:   np.ndarray,
    F_test:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    print(f"\n{'='*65}")
    print(f"  STEP 4 — Feature Scaling (StandardScaler)")
    print(f"{'='*65}")
    
    #Sanity check
    if F_train.ndim != 2:
        raise ValueError(
            f"Expected a 2D feature matrix, got shape {F_train.shape}"
        )

    scaler    = StandardScaler()
    F_train_s = scaler.fit_transform(F_train)   # fit ONLY on train
    F_val_s   = scaler.transform(F_val)
    F_test_s  = scaler.transform(F_test)

    print(f"  Train mean after scaling : {F_train_s.mean():.6f}  (target ≈ 0)")
    print(f"  Train std  after scaling : {F_train_s.std():.6f}   (target ≈ 1)")

    return F_train_s, F_val_s, F_test_s, scaler

# STEP 5 — Logistic Regression Training


def run_lr_training(
    F_train: np.ndarray,
    y_train: np.ndarray,
) -> object:
    print(f"\n{'='*65}")
    print(f"  STEP 5 — Logistic Regression Training")
    print(f"{'='*65}")
    print(f"  Training samples : {len(F_train):,}")
    print(f"  Feature dim      : {F_train.shape[1]:,}")
    print()

    t0       = time.time()
    lr_model = train_lr(F_train, y_train)   # defined in classifier_lr.py
    elapsed  = time.time() - t0

    print(f"\n  LR training complete in {elapsed:.1f}s")
    return lr_model

# STEP 6 — Evaluation

def run_evaluation(
    lr_model,
    F_val:   np.ndarray,
    y_val:   np.ndarray,
    F_test:  np.ndarray,
    y_test:  np.ndarray,
) -> dict:
    print(f"\n{'='*65}")
    print(f"  STEP 6 — Evaluation")
    print(f"{'='*65}")

    print("\n  ── Validation Set ──────────────────────────────────────")
    y_val_pred  = predict_lr(lr_model, F_val)       # classifier_lr.py
    val_metrics = evaluate_model(y_val, y_val_pred)  # evaluation.py

    print("\n  ── Test Set ────────────────────────────────────────────")
    y_test_pred  = predict_lr(lr_model, F_test)
    test_metrics = evaluate_model(y_test, y_test_pred)

    return {"val": val_metrics, "test": test_metrics}


# STEP 7 — Save Artefacts

def save_artefacts(
    cnn: tf.keras.Model,
    lr_model,
    scaler:     StandardScaler,
    metrics:    dict,
    output_dir: str = "saved_model",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  STEP 7 — Saving Artefacts  →  {out.resolve()}")
    print(f"{'='*65}")

    #CNN Model
    cnn_path = out / "cnn.keras"
    cnn.save(cnn_path)
    print(f"  [OK] CNN model       → {cnn_path}")
    
    # LR model — uses save_lr() from classifier_lr.py
    lr_path = out / "lr_model.joblib"
    save_lr(lr_model, str(lr_path))
    print(f"  [OK] LR model        → {lr_path}")

    # Scaler — saved here because scaling is train.py's responsibility
    scaler_path = out / "scaler.joblib"
    joblib.dump(scaler, str(scaler_path))
    print(f"  [OK] StandardScaler  → {scaler_path}")

    # Plain-text report using "classification_report" key from evaluation.py
    report_path = out / "training_report.txt"
    with open(report_path, "w") as f:
        f.write("Diabetic Retinopathy Detection — Training Report\n")
        f.write("=" * 65 + "\n\n")
        for split_name, split_metrics in metrics.items():
            f.write(f"[{split_name.upper()} SET]\n")
            f.write(f"Accuracy : {split_metrics['accuracy']*100:.2f}%\n\n")
            f.write("Classification Report:\n")
            f.write(split_metrics["classification_report"])  # key from evaluation.py
            f.write("\n" + "-" * 65 + "\n\n")

    print(f"  [OK] Text report     → {report_path}")


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diabetic Retinopathy — CNN + Logistic Regression Pipeline"
    )
    parser.add_argument(
        "--data_dir",   default="Processed_Dataset",
        help="Root folder produced by preprocessing.py (default: Processed_Dataset)",
    )
    parser.add_argument(
        "--output_dir", default="saved_model",
        help="Destination for model artefacts (default: saved_model)",
    )
    parser.add_argument(
        "--val_size",   type=float, default=0.15,
        help="Validation fraction (default: 0.15)",
    )
    parser.add_argument(
        "--test_size",  type=float, default=0.15,
        help="Test fraction (default: 0.15)",
    )
    parser.add_argument(
        "--batch_size", type=int,   default=32,
        help="CNN feature-extraction batch size (default: 32)",
    )
    return parser.parse_args()

# MAIN

def main() -> None:
    args = parse_args()

    print("\n" + "=" * 65)
    print("  Diabetic Retinopathy Detection — Training Pipeline")
    print("=" * 65)
    print(f"  Data dir    : {args.data_dir}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Val split   : {args.val_size*100:.0f} %")
    print(f"  Test split  : {args.test_size*100:.0f} %")
    print(f"  Batch size  : {args.batch_size}")

    t_start = time.time()

    # STEP 1: Load image paths + labels
    image_paths, y = load_dataset(args.data_dir)

    # STEP 2: Split paths (NOT images)
    train_paths, val_paths, test_paths, y_train, y_val, y_test = split_dataset(
        image_paths, y,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    # STEP 3: Train CNN (feature extractor)
    cnn = train_cnn(
        train_paths, y_train,
        val_paths, y_val,
        batch_size=args.batch_size
    )

    # STEP 4: Extract features (STREAMING)
    F_train, F_val, F_test, y_train, y_val, y_test = extract_all_features(
        cnn,
        train_paths, val_paths, test_paths,
        y_train, y_val, y_test,
        batch_size=args.batch_size
    )

    # STEP 5: Scale features
    F_train_s, F_val_s, F_test_s, scaler = scale_features(
        F_train, F_val, F_test
    )

    # STEP 6: Train Logistic Regression
    lr_model = run_lr_training(F_train_s, y_train)

    # STEP 7: Evaluate
    metrics = run_evaluation(
        lr_model,
        F_val_s, y_val,
        F_test_s, y_test
    )

    # STEP 8: Save everything
    save_artefacts(
        cnn,
        lr_model,
        scaler,
        metrics,
        output_dir=args.output_dir
    )

    # FINAL SUMMARY
    mins, secs = divmod(int(time.time() - t_start), 60)
    print(f"\n{'='*65}")
    print(f"  PIPELINE COMPLETE  —  {mins}m {secs}s")
    print(f"  Validation accuracy : {metrics['val']['accuracy']*100:.2f}%")
    print(f"  Test accuracy       : {metrics['test']['accuracy']*100:.2f}%")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()