import os
import cv2
import numpy as np
import pandas as pd
import hashlib
import streamlit as st

# ==========================================
# CORE PROCESSING CLASSES (OOP)
# ==========================================

class StorageManager:
    def __init__(self, base_output_dir):
        self.base_dir = base_output_dir
        self.folders = [
            "Resized", "Cropped", "Green_Channel", "Denoised", 
            "Enhanced", "Corrected", "Normalized", "Sharpened", 
            "Augmented", "Final_Output", "Reports"
        ]
        self._create_structure()

    def _create_structure(self):
        for folder in self.folders:
            os.makedirs(os.path.join(self.base_dir, folder), exist_ok=True)

    def save_image(self, img, folder_name, filename):
        path = os.path.join(self.base_dir, folder_name, filename)
        cv2.imwrite(path, img)
        return path

class DatasetLoader:
    SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp')

    @staticmethod
    def get_image_files(directory):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} not found.")
        files = [f for f in os.listdir(directory) if f.lower().endswith(DatasetLoader.SUPPORTED_FORMATS)]
        return [os.path.join(directory, f) for f in files]

class QualityChecker:
    def __init__(self):
        self.hashes = set()

    def check_image(self, filepath, blur_threshold=100.0):
        try:
            img = cv2.imread(filepath)
            if img is None:
                return False, "Corrupted"
            
            with open(filepath, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash in self.hashes:
                return False, "Duplicate"
            self.hashes.add(file_hash)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < blur_threshold:
                return False, f"Blurred ({blur_score:.1f})"

            return True, "Passed"
        except Exception as e:
            return False, f"Error: {str(e)}"

class RetinalPreprocessor:
    def __init__(self, size=(224, 224)):
        self.size = size

    def process(self, img, config):
        results = {}
        
        # 3. Resize
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_CUBIC)
        results['Resized'] = img.copy()

        # 4. Crop Retina
        img = self._crop_retina(img)
        results['Cropped'] = img.copy()

        # 5. Extract Channel
        if config.get('channel') == 'Green':
            img = img[:, :, 1]
        elif config.get('channel') == 'Grayscale':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results['Green_Channel'] = img.copy()

        is_single_channel = len(img.shape) == 2

        # 6. Denoise
        if config.get('denoise') == 'Median':
            img = cv2.medianBlur(img, 3)
        elif config.get('denoise') == 'Gaussian':
            img = cv2.GaussianBlur(img, (5, 5), 0)
        results['Denoised'] = img.copy()

        # 7. Contrast Enhancement (CLAHE)
        if config.get('enhance') == 'CLAHE':
            clahe = cv2.createCLAHE(clipLimit=config.get('clahe_clip', 2.0), tileGridSize=(8,8))
            if is_single_channel:
                img = clahe.apply(img)
            else:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = clahe.apply(l)
                lab = cv2.merge((l, a, b))
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        results['Enhanced'] = img.copy()

        # 8. Illumination Correction
        if config.get('illumination'):
            if is_single_channel:
                bg = cv2.medianBlur(img, 61)
                img = cv2.addWeighted(img, 4, bg, -4, 128)
        results['Corrected'] = img.copy()

        # 10. Sharpening
        if config.get('sharpen') == 'Unsharp Mask':
            gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
            img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
        results['Sharpened'] = img.copy()

        # 9. Normalization
        if config.get('normalize') == 'Min-Max (0-1)':
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        results['Normalized'] = img.copy()
        results['Final_Output'] = img.copy()

        return results

    def _crop_retina(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            return img[y:y+h, x:x+w]
        return img

class DataAugmentor:
    @staticmethod
    def augment(img, config):
        augments = {}
        if config.get('aug_hflip'):
            augments['hflip'] = cv2.flip(img, 1)
        if config.get('aug_vflip'):
            augments['vflip'] = cv2.flip(img, 0)
        if config.get('aug_rotate'):
            rows, cols = img.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
            augments['rot90'] = cv2.warpAffine(img, M, (cols, rows))
        return augments

# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================

st.set_page_config(page_title="Retinal Preprocessing System", layout="wide")
st.title("Retinal Image Preprocessing System")
st.markdown("Automated preprocessing pipeline for Diabetic Retinopathy datasets.")

st.sidebar.header("1. Input/Output Settings")
input_dir = st.sidebar.text_input("Raw Images Folder Path:", "workspace/raw_data")
output_dir = st.sidebar.text_input("Output Folder Path:", "workspace/output")

st.sidebar.header("2. Preprocessing Controls")
img_size = st.sidebar.slider("Resize Resolution", 128, 512, 224)
channel_opt = st.sidebar.selectbox("Color Extraction", ["Green", "Grayscale", "RGB"])
denoise_opt = st.sidebar.selectbox("Noise Removal", ["Median", "Gaussian", "None"])
enhance_opt = st.sidebar.selectbox("Contrast Enhancement", ["CLAHE", "None"])
apply_illum = st.sidebar.checkbox("Apply Illumination Correction", value=True)
sharpen_opt = st.sidebar.selectbox("Sharpening", ["Unsharp Mask", "None"])
norm_opt = st.sidebar.selectbox("Normalization", ["Min-Max (0-1)", "Pixel (0-255)"])

st.sidebar.header("3. Data Augmentation")
aug_hflip = st.sidebar.checkbox("Horizontal Flip")
aug_vflip = st.sidebar.checkbox("Vertical Flip")
aug_rotate = st.sidebar.checkbox("90° Rotation")

config = {
    'channel': channel_opt, 'denoise': denoise_opt, 'enhance': enhance_opt,
    'illumination': apply_illum, 'sharpen': sharpen_opt, 'normalize': norm_opt,
    'aug_hflip': aug_hflip, 'aug_vflip': aug_vflip, 'aug_rotate': aug_rotate,
    'clahe_clip': 2.0
}

if st.button("Start Processing Pipeline", type="primary"):
    if not os.path.exists(input_dir):
        st.error(f"Input directory '{input_dir}' does not exist. Please create it and add raw images.")
        st.stop()

    storage = StorageManager(output_dir)
    checker = QualityChecker()
    preprocessor = RetinalPreprocessor(size=(img_size, img_size))
    
    try:
        files = DatasetLoader.get_image_files(input_dir)
    except Exception as e:
        st.error(str(e))
        st.stop()
        
    total_files = len(files)
    if total_files == 0:
        st.warning("No supported images found in the input directory.")
        st.stop()

    st.info(f"Loaded {total_files} images. Beginning processing...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    log_data = []
    rejected_data = []
    preview_done = False
    col1, col2 = st.columns(2)

    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        status_text.text(f"Processing: {filename} ({i+1}/{total_files})")
        
        passed, msg = checker.check_image(filepath)
        if not passed:
            rejected_data.append({"Filename": filename, "Reason": msg})
            continue

        img = cv2.imread(filepath)
        results = preprocessor.process(img, config)
        
        for step, processed_img in results.items():
            if step == 'Normalized' and norm_opt == 'Min-Max (0-1)':
                save_img = (processed_img * 255).astype('uint8')
            else:
                save_img = processed_img
            storage.save_image(save_img, step, filename)

        augments = DataAugmentor.augment(results['Final_Output'], config)
        for aug_name, aug_img in augments.items():
            aug_filename = f"{os.path.splitext(filename)[0]}_{aug_name}.jpg"
            if norm_opt == 'Min-Max (0-1)':
                 aug_img_save = (aug_img * 255).astype('uint8')
            else:
                 aug_img_save = aug_img
            storage.save_image(aug_img_save, "Augmented", aug_filename)

        log_data.append({"Filename": filename, "Status": "Success", "Augmentations": len(augments)})

        if not preview_done:
            with col1:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
            with col2:
                display_img = results['Final_Output']
                if norm_opt == 'Min-Max (0-1)':
                    display_img = np.clip(display_img, 0, 1)
                if len(display_img.shape) == 2:
                    st.image(display_img, caption="Final Preprocessed", use_column_width=True, clamp=True, channels="GRAY")
                else:
                    st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), caption="Final Preprocessed", use_column_width=True, clamp=True)
            preview_done = True

        progress_bar.progress((i + 1) / total_files)

    status_text.text("Generating Reports...")
    report_dir = os.path.join(output_dir, "Reports")
    if log_data:
        pd.DataFrame(log_data).to_csv(os.path.join(report_dir, "processing_log.csv"), index=False)
    if rejected_data:
        pd.DataFrame(rejected_data).to_csv(os.path.join(report_dir, "rejected_images.csv"), index=False)

    st.markdown("---")
    st.subheader("📊 Processing Summary")
    st.write(f"**Total Images Scanned:** {total_files}")
    st.write(f"**Successfully Processed:** {len(log_data)}")
    st.write(f"**Rejected Images:** {len(rejected_data)}")
    st.success(f"Pipeline Complete! Check output folder: `{output_dir}`")
