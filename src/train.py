import os
import sys
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import cv2
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Suppress TensorFlow info / warning logs (keep errors)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf

# ---------------------------------------------------------------------------
# Import your three existing modules
# ---------------------------------------------------------------------------
# If they live in a sub-package, adjust the import paths accordingly.
from cnn_model    import build_cnn_feature_extractor
from lr_classifier import train_lr, predict_lr, save_lr
from evaluation   import evaluate_model

# ---------------------------------------------------------------------------
# Reproducibility seeds
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SIZE  = (224, 224)
INPUT_SHAPE  = (224, 224, 3)
SUPPORTED    = {".jpg", ".jpeg", ".png", ".bmp"}
BATCH_SIZE   = 32          

CLASS_MAP = {
    "No_DR"            : 0,
    "Mild"             : 1,
    "Moderate"         : 2,
    "Severe"           : 3,
    "Proliferative_DR" : 4,
}

CLASS_NAMES = [
    "No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"
]


# =============================================================================
# STEP 1 — Data Loading
# =============================================================================
def load_image(path: Path) -> np.ndarray | None:
    """
    Read one PNG/JPG from disk, return float32 (224,224,3) in [0,1].
    Returns None if the file is unreadable or has wrong dimensions.
    """
    img = cv2.imread(str(path))
    if img is None:
        print(f"  [WARN] Cannot read: {path.name}")
        return None

    # Converts BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resizes only if the preprocessor somehow missed it
    if img.shape[:2] != TARGET_SIZE:
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

    # Normalizes to [0, 1] float32
    return img.astype(np.float32) / 255.0


def load_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {root.resolve()}\n"
            f"Run preprocessing_system.py first to create it."
        )

    images, labels = [], []
    class_counts   = {name: 0 for name in CLASS_MAP}
    skipped        = 0

    print(f"\n{'='*65}")
    print(f"  STEP 1 — Loading dataset from: {root.resolve()}")
    print(f"{'='*65}")

    for class_name, label in CLASS_MAP.items():
        class_dir = root / class_name
        if not class_dir.is_dir():
            print(f"  [WARN] Class folder missing, skipping: {class_name}")
            continue

        files = [f for f in class_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in SUPPORTED]

        print(f"  {class_name:<22}  {len(files):>5} file(s)")

        for fpath in files:
            img = load_image(fpath)
            if img is None:
                skipped += 1
                continue
            images.append(img)
            labels.append(label)
            class_counts[class_name] += 1

    if not images:
        raise RuntimeError(
            "No images were loaded. Check that Processed_Dataset/ "
            "contains class sub-folders with images."
        )

    X = np.array(images,  dtype=np.float32)   # (N, 224, 224, 3)
    y = np.array(labels,  dtype=np.int32)      # (N,)

    print(f"\n  Total loaded  : {len(X):>6}")
    print(f"  Skipped       : {skipped:>6}")
    print(f"  Shape X       : {X.shape}")
    print(f"  Label range   : {y.min()} – {y.max()}")

    return X, y


# =============================================================================
# STEP 2 — Train / Validation / Test Split
# =============================================================================
def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    val_size:  float = 0.15,
    test_size: float = 0.15,
) -> tuple:
    print(f"\n{'='*65}")
    print(f"  STEP 2 — Splitting dataset")
    print(f"{'='*65}")

    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    # Then split the remaining data into train / val
    # val_size must be rescaled relative to the remaining fraction
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_fraction,
        stratify=y_temp,
        random_state=RANDOM_SEED,
    )

    print(f"  Train        : {len(X_train):>5}  ({len(X_train)/len(X)*100:.1f} %)")
    print(f"  Validation   : {len(X_val):>5}  ({len(X_val)/len(X)*100:.1f} %)")
    print(f"  Test         : {len(X_test):>5}  ({len(X_test)/len(X)*100:.1f} %)")

    # Per-class distribution in train set
    print(f"\n  Train class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls_idx, cnt in zip(unique, counts):
        print(f"    [{cls_idx}] {CLASS_NAMES[cls_idx]:<22}  {cnt:>4}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# STEP 3 — CNN Feature Extraction
# =============================================================================
def extract_features(
    cnn: tf.keras.Model,
    X: np.ndarray,
    batch_size: int = BATCH_SIZE,
    split_name: str = "",
) -> np.ndarray:
    n         = len(X)
    features  = []
    n_batches = (n + batch_size - 1) // batch_size

    tag = f"[{split_name}] " if split_name else ""
    print(f"  {tag}Extracting features from {n} images "
          f"in {n_batches} batch(es) ...")

    for i in range(n_batches):
        start  = i * batch_size
        end    = min(start + batch_size, n)
        batch  = X[start:end]                    # (B, 224, 224, 3)
        feats  = cnn.predict(batch, verbose=0)   # (B, 100352)
        features.append(feats)

        # Simple progress bar
        done = int(30 * end / n)
        bar  = "█" * done + "░" * (30 - done)
        pct  = end / n * 100
        print(f"  {tag}[{bar}] {pct:5.1f}%  ({end}/{n})", end="\r")

    print()  # newline after progress bar
    return np.vstack(features).astype(np.float32)


def extract_all_features(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    print(f"\n{'='*65}")
    print(f"  STEP 3 — CNN Feature Extraction")
    print(f"{'='*65}")

    cnn = build_cnn_feature_extractor(input_shape=INPUT_SHAPE)
    cnn.summary()

    t0 = time.time()
    F_train = extract_features(cnn, X_train, split_name="Train")
    F_val   = extract_features(cnn, X_val,   split_name="Val  ")
    F_test  = extract_features(cnn, X_test,  split_name="Test ")
    elapsed = time.time() - t0

    print(f"\n  Feature extraction complete in {elapsed:.1f}s")
    print(f"  F_train shape : {F_train.shape}")
    print(f"  F_val   shape : {F_val.shape}")
    print(f"  F_test  shape : {F_test.shape}")

    return F_train, F_val, F_test


# =============================================================================
# STEP 4 — Feature Scaling
# =============================================================================
def scale_features(
    F_train: np.ndarray,
    F_val:   np.ndarray,
    F_test:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    
    print(f"\n{'='*65}")
    print(f"  STEP 4 — Feature Scaling (StandardScaler)")
    print(f"{'='*65}")

    scaler    = StandardScaler()
    F_train_s = scaler.fit_transform(F_train)   # fit only on train
    F_val_s   = scaler.transform(F_val)
    F_test_s  = scaler.transform(F_test)

    print(f"  Train mean (post-scale) : {F_train_s.mean():.6f}  (≈ 0)")
    print(f"  Train std  (post-scale) : {F_train_s.std():.6f}   (≈ 1)")

    return F_train_s, F_val_s, F_test_s, scaler


# =============================================================================
# STEP 5 — Logistic Regression Training
# =============================================================================
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

    t0         = time.time()
    lr_model   = train_lr(F_train, y_train)
    elapsed    = time.time() - t0

    print(f"\n  Training complete in {elapsed:.1f}s")
    return lr_model


# =============================================================================
# STEP 6 — Evaluation (Validation + Test)
# =============================================================================
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

    # --- Validation -------------------------------------------------------
    print("\n  ── Validation Set ──────────────────────────────────────")
    y_val_pred = predict_lr(lr_model, F_val)
    val_metrics = evaluate_model(y_val, y_val_pred)

    # --- Test -------------------------------------------------------------
    print("\n  ── Test Set ────────────────────────────────────────────")
    y_test_pred = predict_lr(lr_model, F_test)
    test_metrics = evaluate_model(y_test, y_test_pred)

    return {"val": val_metrics, "test": test_metrics}


# =============================================================================
# STEP 7 — Save Artefacts
# =============================================================================
def save_artefacts(
    lr_model,
    scaler:     StandardScaler,
    metrics:    dict,
    output_dir: str = "saved_model",
) -> None:
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  STEP 7 — Saving Model Artefacts")
    print(f"{'='*65}")

    # LR model
    lr_path = out / "lr_model.joblib"
    save_lr(lr_model, str(lr_path))
    print(f"  [OK] LR model   → {lr_path}")

    # Scaler
    scaler_path = out / "scaler.joblib"
    joblib.dump(scaler, str(scaler_path))
    print(f"  [OK] Scaler     → {scaler_path}")

    # Text report
    report_path = out / "training_report.txt"
    with open(report_path, "w") as f:
        f.write("Diabetic Retinopathy — Training Report\n")
        f.write("=" * 65 + "\n\n")

        for split_name, split_metrics in metrics.items():
            f.write(f"[{split_name.upper()} SET]\n")
            f.write(f"Accuracy : {split_metrics['accuracy']*100:.2f}%\n\n")
            f.write("Classification Report:\n")
            f.write(split_metrics["classification_report"])
            f.write("\n" + "-" * 65 + "\n\n")

    print(f"  [OK] Report     → {report_path}")
    print(f"\n  All artefacts saved to: {out.resolve()}")


# =============================================================================
# MAIN
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DR CNN + Logistic Regression Training Pipeline"
    )
    parser.add_argument(
        "--data_dir",
        default="Processed_Dataset",
        help="Path to the preprocessed image dataset (default: Processed_Dataset)",
    )
    parser.add_argument(
        "--output_dir",
        default="saved_model",
        help="Directory to save trained model artefacts (default: saved_model)",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Fraction of data for validation set (default: 0.15)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Fraction of data for test set (default: 0.15)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for CNN feature extraction (default: 32)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 65)
    print("  Diabetic Retinopathy Detection — Training Pipeline")
    print("=" * 65)
    print(f"  Data dir    : {args.data_dir}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Val split   : {args.val_size*100:.0f} %")
    print(f"  Test split  : {args.test_size*100:.0f} %")
    print(f"  Batch size  : {args.batch_size}")

    pipeline_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load images from Processed_Dataset/
    # ------------------------------------------------------------------
    X, y = load_dataset(args.data_dir)

    # ------------------------------------------------------------------
    # Step 2: Stratified train / val / test split
    # ------------------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    # Free the full array — no longer needed
    del X

    # ------------------------------------------------------------------
    # Step 3: CNN feature extraction (no training — purely forward pass)
    # ------------------------------------------------------------------
    F_train, F_val, F_test = extract_all_features(X_train, X_val, X_test)

    # Free image arrays — features are all we need from here
    del X_train, X_val, X_test

    # ------------------------------------------------------------------
    # Step 4: Scale features (fit on train only)
    # ------------------------------------------------------------------
    F_train_s, F_val_s, F_test_s, scaler = scale_features(
        F_train, F_val, F_test
    )
    del F_train, F_val, F_test  # keep scaled versions only

    # ------------------------------------------------------------------
    # Step 5: Train Logistic Regression
    # ------------------------------------------------------------------
    lr_model = run_lr_training(F_train_s, y_train)

    # ------------------------------------------------------------------
    # Step 6: Evaluate on validation + test sets
    # ------------------------------------------------------------------
    metrics = run_evaluation(
        lr_model,
        F_val_s,  y_val,
        F_test_s, y_test,
    )

    # ------------------------------------------------------------------
    # Step 7: Save model + scaler + report
    # ------------------------------------------------------------------
    save_artefacts(lr_model, scaler, metrics, output_dir=args.output_dir)

    # ------------------------------------------------------------------
    # Final timing
    # ------------------------------------------------------------------
    total_time = time.time() - pipeline_start
    mins, secs = divmod(int(total_time), 60)
    print(f"\n{'='*65}")
    print(f"  PIPELINE COMPLETE  —  Total time: {mins}m {secs}s")
    print(f"  Test accuracy: {metrics['test']['accuracy']*100:.2f}%")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()