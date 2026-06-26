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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info/warning logs
warnings.filterwarnings("ignore")

import tensorflow as tf

<<<<<<< HEAD
from model_cnn     import build_cnn_feature_extractor  # CNN architecture only
from classifier_lr import train_lr, predict_lr, save_lr # LR train/predict/save
from evaluate    import evaluate_model                 # metrics & printing
=======
# ---------------------------------------------------------------------------
# Import three existing modules
# ---------------------------------------------------------------------------
# If they live in a sub-package, adjust the import paths accordingly.
from cnn_model    import build_cnn_feature_extractor
from classifier_lr import train_lr, predict_lr, save_lr
from evaluation   import evaluate_model
>>>>>>> e58da243e0506956ca5c2819f69ac8de8c07bd81

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

TARGET_SIZE = (224, 224)       # (H, W) — matches CNN Input(shape=(224,224,3))
INPUT_SHAPE = (224, 224, 3)    # passed to build_cnn_feature_extractor()
SUPPORTED   = {".jpg", ".jpeg", ".png", ".bmp"}

EXPECTED_FEATURE_DIM = 100352   

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


def load_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {root.resolve()}\n"
            "Run preprocessing.py first to create it."   # fixed filename
        )

    images, labels = [], []
    skipped        = 0

    print(f"\n{'='*65}")
    print(f"  STEP 1 — Loading dataset from: {root.resolve()}")
    print(f"{'='*65}")

    for class_name, label in CLASS_MAP.items():
        class_dir = root / class_name
        if not class_dir.is_dir():
            print(f"  [WARN] Class folder missing — skipping: {class_name}")
            continue

        files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED
        ]
        print(f"  {class_name:<22}  {len(files):>5} file(s)")

        for fpath in files:
            img = load_image(fpath)
            if img is None:
                skipped += 1
                continue
            images.append(img)
            labels.append(label)

    if not images:
        raise RuntimeError(
            "No images were loaded. Verify that Processed_Dataset/ "
            "contains class sub-folders with supported image files."
        )

    X = np.array(images, dtype=np.float32)   # (N, 224, 224, 3)
    y = np.array(labels, dtype=np.int32)      # (N,)

    print(f"\n  Total loaded  : {len(X):>6}")
    print(f"  Skipped       : {skipped:>6}")
    print(f"  Shape X       : {X.shape}")
    print(f"  Label range   : {y.min()} – {y.max()}")

    return X, y


# STEP 2 — Train / Validation / Test Split

def split_dataset(
    X:         np.ndarray,
    y:         np.ndarray,
    val_size:  float = 0.15,
    test_size: float = 0.15,
) -> tuple:
    print(f"\n{'='*65}")
    print(f"  STEP 2 — Splitting dataset")
    print(f"{'='*65}")

    # 1. Test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    # 2. Val split — rescale fraction to the reduced pool
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_fraction,
        stratify=y_temp,
        random_state=RANDOM_SEED,
    )

    n = len(X)
    print(f"  Train      : {len(X_train):>5}  ({len(X_train)/n*100:.1f} %)")
    print(f"  Validation : {len(X_val):>5}  ({len(X_val)/n*100:.1f} %)")
    print(f"  Test       : {len(X_test):>5}  ({len(X_test)/n*100:.1f} %)")

    print(f"\n  Train class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls_idx, cnt in zip(unique, counts):
        print(f"    [{cls_idx}] {CLASS_NAMES[cls_idx]:<22}  {cnt:>4}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# STEP 3 — CNN Feature Extraction


def _extract_split(
    cnn:        tf.keras.Model,
    X:          np.ndarray,
    batch_size: int,
    split_name: str = "",
) -> np.ndarray:
    n         = len(X)
    features  = []
    n_batches = (n + batch_size - 1) // batch_size
    tag       = f"[{split_name}] " if split_name else ""

    print(f"  {tag}Extracting {n} images in {n_batches} batch(es) ...")

    for i in range(n_batches):
        start = i * batch_size
        end   = min(start + batch_size, n)
        feats = cnn.predict(X[start:end], verbose=0)   # (B, 100352)
        features.append(feats)

        done = int(30 * end / n)
        bar  = "█" * done + "░" * (30 - done)
        print(f"  {tag}[{bar}] {end/n*100:5.1f}%  ({end}/{n})", end="\r")

    print()   # newline after progress bar
    return np.vstack(features).astype(np.float32)


def extract_all_features(
    X_train:    np.ndarray,
    X_val:      np.ndarray,
    X_test:     np.ndarray,
    batch_size: int,           # passed from args.batch_size — no silent default
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"\n{'='*65}")
    print(f"  STEP 3 — CNN Feature Extraction")
    print(f"{'='*65}")

    # Build once — reused for all three splits to avoid redundant graph construction
    cnn = build_cnn_feature_extractor(input_shape=INPUT_SHAPE)

    t0      = time.time()
    F_train = _extract_split(cnn, X_train, batch_size, "Train")
    F_val   = _extract_split(cnn, X_val,   batch_size, "Val  ")
    F_test  = _extract_split(cnn, X_test,  batch_size, "Test ")
    
    #Verify CNN output dimension
    assert F_train.shape[1] == EXPECTED_FEATURE_DIM, (
        f"Expected {EXPECTED_FEATURE_DIM} features, got {F_train.shape[1]}"
    )
    assert F_val.shape[1] == EXPECTED_FEATURE_DIM, (
        f"Expected {EXPECTED_FEATURE_DIM} features, got {F_val.shape[1]}"
    )
    assert F_test.shape[1] == EXPECTED_FEATURE_DIM, (
        f"Expected {EXPECTED_FEATURE_DIM} features, got {F_test.shape[1]}"
    )

    elapsed = time.time() - t0  

    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  F_train : {F_train.shape}")
    print(f"  F_val   : {F_val.shape}")
    print(f"  F_test  : {F_test.shape}")

    return F_train, F_val, F_test


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

    #Step 1: Load preprocessed images from disk
    X, y = load_dataset(args.data_dir)

    #Step 2: Stratified 70 / 15 / 15 split (or as configured)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, val_size=args.val_size, test_size=args.test_size,
    )
    del X   # free ~N×224×224×3×4 bytes — no longer needed

    #Step 3: CNN forward pass — pure inference, no CNN training
    F_train, F_val, F_test = extract_all_features(
        X_train, X_val, X_test,
        batch_size=args.batch_size,   # propagated from CLI — not a silent default
    )
    del X_train, X_val, X_test   # free image arrays — features are all we need

    #Step 4: Scale features (fit on train, transform all)
    F_train_s, F_val_s, F_test_s, scaler = scale_features(F_train, F_val, F_test)
    del F_train, F_val, F_test   # keep only scaled versions

    #Step 5: Train Logistic Regression
    lr_model = run_lr_training(F_train_s, y_train)

    #Step 6: Evaluate on validation and test sets
    metrics = run_evaluation(lr_model, F_val_s, y_val, F_test_s, y_test)

    #Step 7: Save LR model, scaler, and text report
    save_artefacts(lr_model, scaler, metrics, output_dir=args.output_dir)

    #Final summary
    mins, secs = divmod(int(time.time() - t_start), 60)
    print(f"\n{'='*65}")
    print(f"  PIPELINE COMPLETE  —  {mins}m {secs}s")
    print(f"  Validation accuracy : {metrics['val']['accuracy']*100:.2f}%")
    print(f"  Test accuracy       : {metrics['test']['accuracy']*100:.2f}%")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()