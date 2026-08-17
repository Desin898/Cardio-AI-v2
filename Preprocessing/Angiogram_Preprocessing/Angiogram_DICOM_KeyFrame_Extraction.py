import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
import pydicom
from datetime import datetime
from skimage.metrics import structural_similarity as ssim


# -----------------------------
# 1. Validation
# -----------------------------
import os
import io
import cv2
import json
import pydicom
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from skimage.metrics import structural_similarity as ssim


# -----------------------------
# 1. Parsing & Normalization
# -----------------------------
def parse_dicom_or_image(file_input):
    """
    Parses raw DICOM files (with or without extensions), multi-frame cine arrays,
    normalizes 12/16-bit intensities to uint8, handles MONOCHROME1 inversion,
    and falls back to OpenCV decoding (cv2.imdecode / cv2.VideoCapture).
    Returns:
        metadata: dict
        frames_8u: 3D uint8 numpy array of shape (N, H, W)
    """
    file_bytes = None
    file_path = None

    if isinstance(file_input, (str, Path)):
        file_path = Path(file_input)
        if file_path.exists():
            try:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
            except Exception:
                pass
    elif isinstance(file_input, bytes):
        file_bytes = file_input
    elif isinstance(file_input, io.BytesIO):
        file_bytes = file_input.getvalue()

    # 1. Try DICOM parsing using pydicom with force=True
    if file_bytes is not None:
        try:
            ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
            if hasattr(ds, "PixelData"):
                pixel_array = ds.pixel_array.astype(np.float32)

                # Invert MONOCHROME1 photometric interpretation so vessels appear dark
                photometric = str(getattr(ds, "PhotometricInterpretation", "")).strip().upper()
                if photometric == "MONOCHROME1":
                    pixel_array = pixel_array.max() - pixel_array

                if pixel_array.ndim == 2:
                    pixel_array = np.expand_dims(pixel_array, axis=0)
                elif pixel_array.ndim == 4:
                    if pixel_array.shape[-1] == 3:
                        grays = [
                            cv2.cvtColor(pixel_array[i].astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
                            for i in range(pixel_array.shape[0])
                        ]
                        pixel_array = np.array(grays)
                    else:
                        pixel_array = pixel_array[..., 0]

                # Robustly normalize 12/16-bit intensities to 8-bit uint8
                frames_8u = []
                for i in range(pixel_array.shape[0]):
                    frame = pixel_array[i]
                    f_min, f_max = frame.min(), frame.max()
                    frame_8u = ((frame - f_min) / (f_max - f_min + 1e-6) * 255.0).astype(np.uint8)
                    frames_8u.append(frame_8u)
                frames_8u = np.array(frames_8u)

                metadata = {
                    "Source": "DICOM",
                    "Modality": str(getattr(ds, "Modality", "XA")),
                    "PhotometricInterpretation": photometric,
                    "NumberOfFrames": frames_8u.shape[0],
                }
                return metadata, frames_8u
        except Exception:
            pass

    # 2. OpenCV Image Fallback (cv2.imdecode)
    if file_bytes is not None:
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            frames_8u = np.expand_dims(img, axis=0)
            return {"Source": "IMAGE", "Modality": "XA (Image)", "NumberOfFrames": 1}, frames_8u

    # 3. OpenCV Video Fallback (cv2.VideoCapture)
    if file_path is not None and file_path.exists():
        cap = cv2.VideoCapture(str(file_path))
        if cap.isOpened():
            frame_list = []
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if frame.ndim == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                frame_list.append(gray.astype(np.uint8))
            cap.release()
            if frame_list:
                frames_8u = np.array(frame_list)
                return {"Source": "MP4", "Modality": "XA (Simulated)", "NumberOfFrames": frames_8u.shape[0]}, frames_8u

    raise ValueError("Unsupported or unreadable angiogram file format.")


def validate_input(file_path: Path) -> bool:
    try:
        _, frames = parse_dicom_or_image(file_path)
        return len(frames) > 0
    except Exception:
        return False


def read_angiogram(file_path: Path):
    return parse_dicom_or_image(file_path)


# -----------------------------
# 2. Frame Quality & Selection
# -----------------------------
def compute_frame_quality_metrics(frame: np.ndarray):
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    mean_intensity = float(np.mean(gray))
    contrast = float(np.std(gray))

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = float(np.mean(np.sqrt(sobel_x**2 + sobel_y**2)))

    noise = float(np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0)))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "mean_intensity": mean_intensity,
        "contrast": contrast,
        "edge_strength": edge_strength,
        "noise": noise,
        "gradient_sharpness": laplacian_var,
        "quality_score": contrast * laplacian_var,
    }


def compute_sharpness(frame: np.ndarray) -> float:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(frame, cv2.CV_64F).var())


def select_best_frames(frames: np.ndarray, top_k=3):
    """Selects frame(s) with highest contrast/gradient sharpness score."""
    scores = []
    for f in frames:
        m = compute_frame_quality_metrics(f)
        scores.append(m["quality_score"])
    sorted_indices = np.argsort(scores)[::-1]
    return sorted_indices[:top_k]


def process_angiogram(file_path: Path, output_root: str = "output"):
    file_path = Path(file_path)
    patient_id = file_path.stem
    patient_output_dir = Path(output_root) / patient_id
    patient_output_dir.mkdir(parents=True, exist_ok=True)

    metadata, frames = parse_dicom_or_image(file_path)

    if frames.shape[0] == 1:
        selected_indices = [0]
    else:
        selected_indices = select_best_frames(frames, top_k=min(3, frames.shape[0]))

    variants = []
    for rank, idx in enumerate(selected_indices):
        norm = frames[idx]
        filename = f"{patient_id}_frame_{idx}_rank_{rank + 1}.png"
        filepath = patient_output_dir / filename
        cv2.imwrite(str(filepath), norm)

        metrics = compute_frame_quality_metrics(norm)
        variants.append({
            "rank": rank + 1,
            "frame_index": int(idx),
            "filename": filename,
            "path": str(filepath),
            "label": f"Keyframe #{rank + 1} (Frame {idx})",
            "sharpness": float(metrics["gradient_sharpness"]),
            "contrast": float(metrics["contrast"]),
            "edge_strength": float(metrics["edge_strength"]),
            "quality_score": float(metrics["quality_score"]),
        })

    pipeline_metadata = {
        "patient_id": patient_id,
        "processed_at": datetime.now().isoformat(),
        "source": metadata.get("Source", "Unknown"),
        "number_of_original_frames": int(metadata.get("NumberOfFrames", 1)),
        "selected_frame_indices": [int(i) for i in selected_indices],
        "output_directory": str(patient_output_dir),
        "variants": variants,
    }

    metadata_path = patient_output_dir / "pipeline_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(pipeline_metadata, f, indent=4)

    return pipeline_metadata