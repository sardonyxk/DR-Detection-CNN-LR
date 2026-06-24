"""
=============================================================================
Diabetic Retinopathy Detection System
Retinal Fundus Image Preprocessing Pipeline
=============================================================================
Pipeline:
  Raw Image → Validation → Resize (224×224) → RGB Conversion
           → Normalization (0–1, float32) → Augmentation → Save

Supported formats : JPG, JPEG, PNG, BMP
Output layout     : mirrors input class-folder structure
Dependencies      : opencv-python, numpy (both ship with most ML environments)
=============================================================================
"""

import os
import sys
import warnings
import time
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")  # suppress low-level codec warnings

# ---------------------------------------------------------------------------
# Global configuration – tweak these without touching function logic
# ---------------------------------------------------------------------------
TARGET_SIZE       = (224, 224)          # CNN input dimensions (H × W)
SUPPORTED_EXTS    = {".jpg", ".jpeg", ".png", ".bmp"}
INTERPOLATION     = cv2.INTER_LANCZOS4  # high-quality downsampling

# Augmentation hyper-parameters (kept mild to preserve retinal features)
ROTATION_ANGLES   = [15, -15]           # degrees
ZOOM_FACTOR       = 0.10                # ±10 % random zoom
BRIGHTNESS_RANGE  = (0.80, 1.20)        # multiplicative brightness range
SHIFT_RANGE       = 0.05               # fraction of image dimension


# ---------------------------------------------------------------------------
# Step 1 – Image Validation
# ---------------------------------------------------------------------------
def validate_image(image_path: str) -> bool:
    """
    Check that a file exists, has a supported extension, and can be decoded.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the candidate image file.

    Returns
    -------
    bool
        True  → file is a valid, readable image.
        False → file is missing, unsupported, or corrupted.
    """
    path = Path(image_path)

    # 1a. Extension check
    if path.suffix.lower() not in SUPPORTED_EXTS:
        print(f"  [SKIP] Unsupported format: {path.name}")
        return False

    # 1b. Existence check
    if not path.is_file():
        print(f"  [SKIP] File not found: {image_path}")
        return False

    # 1c. Decode check – cv2.imread silently returns None for corrupt files
    img = cv2.imread(str(path))
    if img is None:
        print(f"  [WARN] Corrupted / unreadable image skipped: {path.name}")
        return False

    # 1d. Sanity-check minimum dimensions
    if img.shape[0] < 10 or img.shape[1] < 10:
        print(f"  [WARN] Image too small (degenerate file): {path.name}")
        return False

    return True


# ---------------------------------------------------------------------------
# Step 2-4 – Core preprocessing (resize → RGB → normalize)
# ---------------------------------------------------------------------------
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Apply the core preprocessing pipeline to a single image array.

    Steps performed
    ---------------
    1. Resize  : to TARGET_SIZE using Lanczos interpolation.
    2. RGB     : convert BGR (OpenCV default) → RGB; handle grayscale inputs.
    3. Normalize: cast to float32 and scale pixel values to [0.0, 1.0].

    Parameters
    ----------
    image : np.ndarray
        Raw image array as loaded by cv2.imread (BGR, uint8).

    Returns
    -------
    np.ndarray
        Processed image: shape (224, 224, 3), dtype float32, values in [0, 1].
    """

    # --- Step 2: Resize ---------------------------------------------------
    # INTER_LANCZOS4 preserves fine retinal vessel details better than
    # bilinear or nearest-neighbour resizing.
    resized = cv2.resize(image, TARGET_SIZE, interpolation=INTERPOLATION)

    # --- Step 3: RGB conversion -------------------------------------------
    if len(resized.shape) == 2:
        # Grayscale → stack into 3-channel image
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    elif resized.shape[2] == 4:
        # RGBA → RGB (drop alpha channel)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
    else:
        # Standard BGR → RGB
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Confirm exactly 3 channels after conversion
    assert resized.shape == (TARGET_SIZE[0], TARGET_SIZE[1], 3), (
        f"Unexpected shape after RGB conversion: {resized.shape}"
    )

    # --- Step 4: Normalization --------------------------------------------
    # Cast to float32 first to avoid integer overflow during division.
    normalized = resized.astype(np.float32) / 255.0

    return normalized  # shape: (224, 224, 3), dtype: float32, range: [0, 1]


# ---------------------------------------------------------------------------
# Step 5 – Data Augmentation
# ---------------------------------------------------------------------------
def _float_to_uint8(img: np.ndarray) -> np.ndarray:
    """Clip a float32 [0,1] image to uint8 [0,255] for saving with OpenCV."""
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def augment_image(image: np.ndarray) -> dict:
    """
    Generate a set of augmented variants from a preprocessed image.

    All transformations are deliberately conservative so that diagnostic
    features of retinal fundus images (optic disc, blood vessels, lesions)
    remain intact and recognisable.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed image: float32, shape (224, 224, 3), values in [0, 1].

    Returns
    -------
    dict
        Mapping of suffix → augmented image (same dtype/shape as input).
        Keys: 'flip', 'rot15', 'rot-15', 'zoom', 'bright', 'shift'
    """
    h, w = image.shape[:2]
    augmented = {}

    # ------------------------------------------------------------------ #
    # A) Horizontal Flip                                                   #
    # Retinal images are horizontally symmetric between left/right eyes.   #
    # ------------------------------------------------------------------ #
    augmented["flip"] = np.fliplr(image).copy()

    # ------------------------------------------------------------------ #
    # B) Rotation  (+15° and −15°)                                        #
    # Rotation centre = image centre; no translation; same crop size.     #
    # ------------------------------------------------------------------ #
    centre = (w / 2.0, h / 2.0)
    for angle in ROTATION_ANGLES:
        M = cv2.getRotationMatrix2D(centre, angle, scale=1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101  # reflect to avoid black borders
        )
        suffix = f"rot{angle:+d}".replace("+", "")   # 'rot15' / 'rot-15'
        augmented[suffix] = rotated

    # ------------------------------------------------------------------ #
    # C) Random Zoom (up to ±10 %)                                        #
    # Crop a central region then resize back to TARGET_SIZE.              #
    # ------------------------------------------------------------------ #
    zoom = 1.0 + np.random.uniform(-ZOOM_FACTOR, ZOOM_FACTOR)
    new_h = int(h * zoom)
    new_w = int(w * zoom)
    zoomed = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    if zoom > 1.0:
        # Crop centre to original size
        y_start = (new_h - h) // 2
        x_start = (new_w - w) // 2
        zoomed = zoomed[y_start:y_start + h, x_start:x_start + w]
    else:
        # Pad with border reflection then crop
        pad_y = (h - new_h) // 2
        pad_x = (w - new_w) // 2
        zoomed = cv2.copyMakeBorder(
            zoomed, pad_y, h - new_h - pad_y,
            pad_x, w - new_w - pad_x,
            cv2.BORDER_REFLECT_101
        )
        zoomed = zoomed[:h, :w]  # safety crop to exact size

    augmented["zoom"] = zoomed

    # ------------------------------------------------------------------ #
    # D) Brightness Adjustment                                            #
    # Multiply pixel values by a random scalar in BRIGHTNESS_RANGE.      #
    # Clipping to [0,1] prevents overflow artifacts.                      #
    # ------------------------------------------------------------------ #
    factor = np.random.uniform(*BRIGHTNESS_RANGE)
    bright = np.clip(image * factor, 0.0, 1.0).astype(np.float32)
    augmented["bright"] = bright

    # ------------------------------------------------------------------ #
    # E) Small Width & Height Shifts (≤ 5 % of dimension)                #
    # Translation matrix keeps content visible; borders are reflected.   #
    # ------------------------------------------------------------------ #
    tx = np.random.uniform(-SHIFT_RANGE, SHIFT_RANGE) * w  # pixels
    ty = np.random.uniform(-SHIFT_RANGE, SHIFT_RANGE) * h
    M_shift = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(
        image, M_shift, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101
    )
    augmented["shift"] = shifted

    return augmented


# ---------------------------------------------------------------------------
# Step 6 – Save helper
# ---------------------------------------------------------------------------
def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save a float32 [0,1] image to disk as a PNG (lossless).

    Parameters
    ----------
    image       : float32 ndarray, values in [0, 1], shape (H, W, 3).
    output_path : destination file path (directories are created if absent).

    Returns
    -------
    bool : True on success, False on failure.
    """
    try:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Convert back to uint8 BGR for OpenCV imwrite
        img_uint8 = _float_to_uint8(image)
        img_bgr   = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        # PNG for lossless quality; change to ".jpg" + quality param if space matters
        success = cv2.imwrite(str(out), img_bgr)
        if not success:
            print(f"  [ERROR] cv2.imwrite failed for: {output_path}")
            return False
        return True

    except Exception as exc:
        print(f"  [ERROR] Could not save {output_path}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Master pipeline – process entire dataset
# ---------------------------------------------------------------------------
def process_dataset(input_folder: str, output_folder: str) -> dict:
    """
    Walk *input_folder*, preprocess every valid retinal image, generate
    augmented variants, and write results to *output_folder* mirroring the
    original class-folder hierarchy.

    Parameters
    ----------
    input_folder  : Root directory of the raw dataset.
    output_folder : Root directory for the processed dataset.

    Returns
    -------
    dict with summary statistics:
        total_found      – files discovered with supported extensions
        processed_ok     – images successfully preprocessed
        augmented_saved  – augmented variants written to disk
        corrupted_skip   – files that failed validation
        save_errors      – images that passed preprocessing but failed on save
    """
    input_root  = Path(input_folder)
    output_root = Path(output_folder)

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    output_root.mkdir(parents=True, exist_ok=True)

    # Counters
    stats = {
        "total_found"    : 0,
        "processed_ok"   : 0,
        "augmented_saved": 0,
        "corrupted_skip" : 0,
        "save_errors"    : 0,
    }

    # Collect all candidate files (recursive walk)
    all_files = [
        p for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    stats["total_found"] = len(all_files)

    if stats["total_found"] == 0:
        print(f"\n[WARNING] No supported images found under: {input_folder}")
        return stats

    print(f"\n{'='*60}")
    print(f"  Diabetic Retinopathy – Image Preprocessing Pipeline")
    print(f"{'='*60}")
    print(f"  Input  : {input_root.resolve()}")
    print(f"  Output : {output_root.resolve()}")
    print(f"  Images : {stats['total_found']} files discovered")
    print(f"{'='*60}\n")

    start_time = time.time()

    for idx, img_path in enumerate(all_files, start=1):

        # Progress indicator every 10 images (or always for small datasets)
        if idx % 10 == 0 or idx == 1 or idx == stats["total_found"]:
            pct = idx / stats["total_found"] * 100
            print(f"  [{idx:>5}/{stats['total_found']}]  {pct:5.1f}%  {img_path.name}")

        # ------------------------------------------------------------------
        # Step 1: Validate
        # ------------------------------------------------------------------
        if not validate_image(str(img_path)):
            stats["corrupted_skip"] += 1
            continue

        # ------------------------------------------------------------------
        # Load raw image (BGR uint8)
        # ------------------------------------------------------------------
        raw = cv2.imread(str(img_path))
        if raw is None:
            print(f"  [WARN] cv2.imread returned None: {img_path.name}")
            stats["corrupted_skip"] += 1
            continue

        # ------------------------------------------------------------------
        # Steps 2–4: Preprocess
        # ------------------------------------------------------------------
        try:
            processed = preprocess_image(raw)
        except Exception as exc:
            print(f"  [ERROR] Preprocessing failed for {img_path.name}: {exc}")
            stats["corrupted_skip"] += 1
            continue

        # ------------------------------------------------------------------
        # Build mirrored output path (preserve class-folder structure)
        # relative_path = class_folder / filename
        # ------------------------------------------------------------------
        relative = img_path.relative_to(input_root)
        stem     = img_path.stem              # filename without extension
        out_dir  = output_root / relative.parent  # e.g. Processed/Mild/

        # ------------------------------------------------------------------
        # Step 6a: Save original processed image
        # ------------------------------------------------------------------
        orig_out = out_dir / f"{stem}.png"
        if save_image(processed, str(orig_out)):
            stats["processed_ok"] += 1
        else:
            stats["save_errors"] += 1

        # ------------------------------------------------------------------
        # Step 5: Augment
        # ------------------------------------------------------------------
        try:
            augmented_variants = augment_image(processed)
        except Exception as exc:
            print(f"  [ERROR] Augmentation failed for {img_path.name}: {exc}")
            continue

        # ------------------------------------------------------------------
        # Step 6b: Save each augmented variant
        # Naming convention:  <stem>_<suffix>.png
        # e.g.  image01_flip.png, image01_rot15.png, …
        # ------------------------------------------------------------------
        for suffix, aug_img in augmented_variants.items():
            aug_out = out_dir / f"{stem}_{suffix}.png"
            if save_image(aug_img, str(aug_out)):
                stats["augmented_saved"] += 1
            else:
                stats["save_errors"] += 1

    elapsed = time.time() - start_time

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  PROCESSING COMPLETE  ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  Total images found       : {stats['total_found']:>6}")
    print(f"  Successfully processed   : {stats['processed_ok']:>6}")
    print(f"  Augmented images saved   : {stats['augmented_saved']:>6}  "
          f"({len(augmented_variants)} variants × {stats['processed_ok']} images)")
    print(f"  Corrupted / skipped      : {stats['corrupted_skip']:>6}")
    print(f"  Save errors              : {stats['save_errors']:>6}")
    total_output = stats["processed_ok"] + stats["augmented_saved"]
    print(f"  Total output files       : {total_output:>6}")
    print(f"{'='*60}\n")

    return stats


# ---------------------------------------------------------------------------
# Utility: quick single-image preview (useful during development)
# ---------------------------------------------------------------------------
def preprocess_single(image_path: str, output_path: str) -> bool:
    """
    Convenience wrapper: validate, preprocess, and save ONE image.
    No augmentation is applied – useful for inspecting pipeline output.

    Parameters
    ----------
    image_path  : path to a raw retinal image.
    output_path : where to save the processed result.

    Returns
    -------
    bool : True on success.
    """
    if not validate_image(image_path):
        return False

    raw = cv2.imread(image_path)
    if raw is None:
        print(f"[ERROR] Cannot read: {image_path}")
        return False

    processed = preprocess_image(raw)
    saved     = save_image(processed, output_path)

    if saved:
        print(f"[OK] Saved preprocessed image → {output_path}")
        print(f"     Shape : {processed.shape}")
        print(f"     dtype : {processed.dtype}")
        print(f"     Range : [{processed.min():.4f}, {processed.max():.4f}]")
    return saved


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # CONFIGURATION – edit these two paths before running
    # -----------------------------------------------------------------------
    INPUT_FOLDER  = "Dataset"            # folder with raw fundus images
    OUTPUT_FOLDER = "Processed_Dataset"  # destination for processed images
    # -----------------------------------------------------------------------

    # Allow overriding via command-line arguments for convenience:
    #   python retinal_preprocessing.py <input_folder> <output_folder>
    if len(sys.argv) == 3:
        INPUT_FOLDER  = sys.argv[1]
        OUTPUT_FOLDER = sys.argv[2]
    elif len(sys.argv) != 1:
        print("Usage: python retinal_preprocessing.py [input_folder output_folder]")
        sys.exit(1)

    # Run the full pipeline
    summary = process_dataset(INPUT_FOLDER, OUTPUT_FOLDER)

    # Non-zero exit code when nothing was processed (useful in CI/CD scripts)
    if summary["processed_ok"] == 0:
        sys.exit(1)