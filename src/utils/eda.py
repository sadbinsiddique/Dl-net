import os
import glob
import hashlib
import urllib.request
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageStat
import imagehash
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from skimage.metrics import structural_similarity as ssim
import dotenv
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

TARGET_SIZE = (128, 128)
DEFAULT_FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

dotenv.load_dotenv()

try:
    from importlib import import_module
    umap = import_module("umap")
    UMAP_AVAILABLE = True
except ImportError:
    umap = None
    UMAP_AVAILABLE = False

MP_AVAILABLE = True

warnings.filterwarnings('ignore')

class ImageDatasetEDA:
    def __init__(self, base_path):
        self.base_path = base_path
        self.filepaths = glob.glob(os.path.join(base_path, '**/*.*'), recursive=True)
        self.metadata = []
        self.corrupted = []
        self.image_hashes = defaultdict(list)

    def validate_and_extract_metadata(self):
        """Scans for structure, extensions, modes, and corruption."""
        print(f"[2]Found {len(self.filepaths)} total files.")

        for path in tqdm(self.filepaths, desc="> Extracting Metadata"):
            ext = os.path.splitext(path)[1].lower()
            folder = os.path.basename(os.path.dirname(path))

            try:
                with Image.open(path) as img:
                    img.verify()  # Fast corrupted check

                # Reopen to get details (verify() closes the file)
                with Image.open(path) as img:
                    w, h = img.size
                    mode = img.mode

                    self.metadata.append({
                        'filepath': path,
                        'class': folder,
                        'extension': ext,
                        'width': w,
                        'height': h,
                        'resolution': w * h,
                        'mode': mode,
                        'aspect_ratio': round(w / h, 3)
                    })
            except Exception as e:
                self.corrupted.append({'filepath': path, 'error': str(e)})

        self.df = pd.DataFrame(self.metadata)
        print(f"[4]Valid images: {len(self.df)} | Corrupted: {len(self.corrupted)}")

    def analyze_distribution(self):
        """Class distribution and balance ratio."""
        dist = self.df['class'].value_counts()
        balance_ratio = dist.min() / dist.max()

        plt.figure(figsize=(12, 5))
        sns.barplot(x=dist.index, y=dist.values, palette='viridis')
        plt.title(f'Class Distribution (Balance Ratio: {balance_ratio:.2f})')
        plt.xticks(rotation=45)
        plt.show()

        return dist, balance_ratio


def compute_image_metrics(row):
    """Calculates brightness, contrast, blur, sharpness, noise, and entropy."""
    path = row['filepath']
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None: return pd.Series(dtype='float64')

    # Blur / Sharpness (Laplacian Variance)
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()

    # Brightness & Contrast (Mean and Std of pixel intensities)
    brightness = np.mean(img_gray)
    contrast = np.std(img_gray)

    # Edge Density (Canny)
    edges = cv2.Canny(img_gray, 100, 200)
    edge_density = np.sum(edges > 0) / (img_gray.shape[0] * img_gray.shape[1])

    # Entropy (Information content)
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-7))

    # Noise Estimation (Median Blur residual)
    median = cv2.medianBlur(img_gray, 3)
    noise = np.std(img_gray - median)

    return pd.Series([brightness, contrast, laplacian_var, edge_density, entropy, noise],
                     index=['brightness', 'contrast', 'sharpness', 'edge_density', 'entropy', 'noise'])

def run_quality_pipeline(df):
    print("[5]Calculating image metrics...")
    # Remove tqdm.pandas()
    # Use standard .apply() instead of .progress_apply()
    metrics_df = df.apply(compute_image_metrics, axis=1)
    return pd.concat([df, metrics_df], axis=1)


def find_duplicates_and_leakage(df):
    """Uses perceptual hashing to find exact and near-duplicates."""
    hashes = {}
    duplicates = []

    for path in tqdm(df['filepath'], desc="Generating pHashes"):
        try:
            img = Image.open(path)
            h = str(imagehash.phash(img))
            if h in hashes:
                duplicates.append((path, hashes[h]))
            else:
                hashes[h] = path
        except:
            continue

    print(f"Found {len(duplicates)} potential duplicates or leaked images.")
    return duplicates

def calculate_ssim_outliers(df, sample_size=500):
    """Calculates SSIM against a mean face to find heavy outliers."""
    sampled = df.sample(min(sample_size, len(df)))
    images = []
    for p in sampled['filepath']:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(cv2.resize(img, TARGET_SIZE))

    mean_img = np.mean(images, axis=0).astype(np.uint8)
    ssim_scores = [ssim(img, mean_img, data_range=img.max() - img.min()) for img in images]

    sampled['ssim_to_mean'] = ssim_scores
    outliers = sampled[sampled['ssim_to_mean'] < sampled['ssim_to_mean'].quantile(0.05)]

    return mean_img, outliers


def visualize_embeddings(df, sample_size=1000):
    """Flattens images and runs PCA, t-SNE, and Isolation Forest."""
    sampled = df.sample(min(sample_size, len(df)))
    labels = sampled['class']

    print("[6]Flattening images for PCA, t-SNE, and Isolation Forest")
    flat_images = []
    for p in tqdm(sampled['filepath']):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # skip unreadable images
            continue
        img = cv2.resize(img, TARGET_SIZE)
        flat_images.append(img.flatten())

    X = np.array(flat_images)

    # 1. PCA
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X)

    # 2. t-SNE
    print("[7]Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_pca)

    # 3. Outlier Detection (Isolation Forest)
    iso = IsolationForest(contamination=0.05, random_state=42)
    outliers = iso.fit_predict(X_pca)

    plt.figure(figsize=(16, 6))

    # t-SNE Plot
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette='tab10', s=30)
    plt.title("t-SNE Projection of Image Pixels")

    # Outlier Plot
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=outliers, palette={1: 'blue', -1: 'red'}, s=30)
    plt.title("Isolation Forest Outliers (Red)")

    plt.tight_layout()
    plt.savefig("embeddings_outliers.png", dpi=300)
    plt.show()


def _resolve_face_landmarker_model():
    model_path = os.getenv("MEDIAPIPE_FACE_LANDMARKER_MODEL")
    if model_path:
        resolved_path = Path(model_path).expanduser()
        if resolved_path.exists():
            return resolved_path

    cache_dir = Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache") / "dl-net"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_model = cache_dir / "face_landmarker.task"

    if not cached_model.exists():
        try:
            urllib.request.urlretrieve(DEFAULT_FACE_LANDMARKER_MODEL_URL, cached_model)
        except Exception:
            return None

    return cached_model


def analyze_faces_mediapipe(df, sample_size=500):
    """Detects faces and extracts 468 facial landmarks."""
    if not MP_AVAILABLE:
        print("MediaPipe is not installed. Skipping landmark detection.")
        return pd.DataFrame()

    try:
        model_path = _resolve_face_landmarker_model()
        if model_path is None:
            print("MediaPipe face landmarker model is not available. Run scripts/download_mediapipe_face_landmarker.py or set MEDIAPIPE_FACE_LANDMARKER_MODEL.")
            return pd.DataFrame()

        sampled = df.sample(min(sample_size, len(df)))
        landmark_rows = []
        detected_count = 0

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        with FaceLandmarker.create_from_options(options) as face_landmarker:
            for path in tqdm(sampled['filepath'], desc="Detecting Facial Landmarks"):
                image = cv2.imread(path)
                if image is None:
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

                results = face_landmarker.detect(mp_image)
                if not results.face_landmarks:
                    continue

                detected_count += 1
                face_landmarks = results.face_landmarks[0]
                row = {'filepath': path, 'landmarks_detected': len(face_landmarks)}

                for idx, landmark in enumerate(face_landmarks):
                    row[f'x_{idx}'] = landmark.x
                    row[f'y_{idx}'] = landmark.y
                    row[f'z_{idx}'] = landmark.z

                landmark_rows.append(row)

        landmarks_df = pd.DataFrame(landmark_rows)
        print(f"Faces detected via MediaPipe in {detected_count}/{len(sampled)} sampled images.")
        return landmarks_df

    except Exception as e:
        print(f"MediaPipe initialization failed with error: {e}. Skipping landmark extraction.")
        return pd.DataFrame()
