import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter

from backend.app.core.config import settings
from Preprocessing.Angiogram_Preprocessing.Angiogram_DICOM_KeyFrame_Extraction import parse_dicom_or_image


class QCAService:
    """
    Quantitative Coronary Angiography (QCA) Service.
    Extracts vessel centerlines via Medial Axis Transform / Skeletonization,
    profiles perpendicular lumen diameters, calculates percentage stenosis,
    grades lesion severity, and renders high-contrast annotated diagnostic images.
    """

    @staticmethod
    def isolate_arterial_tree(binary_mask: np.ndarray, min_area: int = 400) -> np.ndarray:
        """
        Applies Connected Component Analysis (cv2.connectedComponentsWithStats)
        to suppress noise blobs and isolate strictly the primary connected arterial tree.
        Applies morphological closing to smooth vessel borders and close lumen gaps.
        """
        if binary_mask is None or np.count_nonzero(binary_mask) == 0:
            return np.zeros_like(binary_mask, dtype=np.uint8)

        # 1. Connected Component Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask.astype(np.uint8), connectivity=8
        )

        cleaned_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        if num_labels <= 1:
            return cleaned_mask

        # Filter components (label 0 is background)
        valid_labels = []
        areas = stats[1:, cv2.CC_STAT_AREA]

        # Keep components with area > min_area, or fall back to largest component if none > min_area
        large_indices = np.where(areas >= min_area)[0] + 1
        if len(large_indices) > 0:
            valid_labels = list(large_indices)
        else:
            largest_label = int(np.argmax(areas)) + 1
            valid_labels = [largest_label]

        for label_idx in valid_labels:
            cleaned_mask[labels == label_idx] = 255

        # 2. Morphological Closing with Elliptical Kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        return closed_mask

    @classmethod
    def preprocess_mask(cls, mask_input: Any) -> np.ndarray:
        """
        Converts mask or raw image input into a clean binary uint8 vessel mask.
        Enhances contrast using CLAHE, applies adaptive thresholding/segmentation,
        suppresses noise via Connected Component Analysis, and closes lumen gaps.
        """
        if isinstance(mask_input, (str, Path)):
            path_obj = Path(mask_input)
            if not path_obj.exists():
                raise FileNotFoundError(f"Mask file not found: {mask_input}")
            try:
                _, frames = parse_dicom_or_image(path_obj)
                mask_np = frames[0]
            except Exception:
                mask_np = cv2.imread(str(mask_input), cv2.IMREAD_GRAYSCALE)
        elif isinstance(mask_input, Image.Image):
            mask_np = np.array(mask_input.convert("L"))
        elif isinstance(mask_input, np.ndarray):
            mask_np = mask_input.copy()
            if mask_np.ndim == 3:
                mask_np = cv2.cvtColor(mask_np, cv2.COLOR_BGR2GRAY)
        else:
            raise TypeError("Unsupported mask_input format. Expected PIL.Image, np.ndarray, or file path.")

        # Determine if input is a binary mask or grayscale raw image
        unique_vals = np.unique(mask_np)
        if len(unique_vals) <= 2:
            binary_mask = ((mask_np > 127).astype(np.uint8)) * 255
        else:
            # Contrast Enhancement using CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(mask_np)

            # Adaptive Thresholding / Segmentation
            blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
            binary_mask = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 3
            )

        # Isolate arterial tree & apply morphological closing
        return cls.isolate_arterial_tree(binary_mask, min_area=400)

    @staticmethod
    def extract_centerline_and_diameters(binary_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts vessel centerline coordinates and Euclidean Distance Transform radii.
        Returns:
            skeleton: 2D boolean array of centerline pixels
            distance_map: 2D float array of perpendicular distances to nearest boundary
            y_coords, x_coords: 1D arrays of ordered centerline pixel coordinates
        """
        binary_bool = binary_mask > 0
        skeleton = skeletonize(binary_bool)
        distance_map = distance_transform_edt(binary_bool)

        y_coords, x_coords = np.where(skeleton)
        if len(x_coords) > 2:
            # Order skeleton coordinates sequentially along the vessel path
            points = np.column_stack((y_coords, x_coords))
            start_idx = np.argmin(points[:, 0] + points[:, 1])

            ordered_indices = [start_idx]
            visited = {start_idx}
            curr = start_idx

            while len(visited) < len(points):
                curr_pt = points[curr]
                unvisited = [i for i in range(len(points)) if i not in visited]
                dists = np.sum((points[unvisited] - curr_pt) ** 2, axis=1)
                best_u = np.argmin(dists)
                next_idx = unvisited[best_u]

                visited.add(next_idx)
                ordered_indices.append(next_idx)
                curr = next_idx

            y_coords = y_coords[ordered_indices]
            x_coords = x_coords[ordered_indices]

        return skeleton, distance_map, y_coords, x_coords

    @staticmethod
    def _smooth_profile(profile: np.ndarray) -> np.ndarray:
        """Smooths diameter profile using Savitzky-Golay filter or moving average."""
        n = len(profile)
        if n < 5:
            return profile.copy()
        win = min(9, n if n % 2 == 1 else n - 1)
        if win >= 5:
            try:
                return savgol_filter(profile, window_length=win, polyorder=2, mode="interp")
            except Exception:
                pass
        return np.convolve(profile, np.ones(5) / 5, mode="same")

    def compute_qca_metrics(
        self,
        binary_mask: np.ndarray,
        pixel_spacing_mm: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculates QCA stenosis metrics:
          - Minimum Lumen Diameter (d_min)
          - Reference Vessel Diameter (d_ref)
          - Percentage Stenosis = (1 - (d_min / d_ref)) * 100
          - Lesion Severity: MILD (<50%), MODERATE (50-69%), SEVERE (>=70%)
          - Intervention Recommended (True for SEVERE)
        """
        skeleton, distance_map, y_coords, x_coords = self.extract_centerline_and_diameters(binary_mask)

        if len(x_coords) == 0:
            return {
                "stenosis_percentage": 0.0,
                "severity_grade": "MILD",
                "d_min": 10.0,
                "d_ref": 10.0,
                "lesion_coordinates": {"x": 0, "y": 0},
                "intervention_recommended": False,
                "centerline_points_count": 0,
                "centerline_coords": [],
            }

        radii = distance_map[y_coords, x_coords]
        diameters = radii * 2.0
        smooth_diameters = self._smooth_profile(diameters)
        n_points = len(smooth_diameters)

        # Ignore vessel endpoints / tapering tips (ignore first and last 10% of skeleton points)
        trim = int(np.floor(0.10 * n_points))
        if n_points > 2 * trim + 3 and trim > 0:
            valid_indices = list(range(trim, n_points - trim))
        else:
            valid_indices = list(range(n_points))

        valid_diameters = smooth_diameters[valid_indices]

        # Identify bottleneck minimum lumen diameter (d_min)
        min_local_idx = int(np.argmin(valid_diameters))
        min_idx = valid_indices[min_local_idx]

        d_min = float(smooth_diameters[min_idx])
        x_min = int(x_coords[min_idx])
        y_min = int(y_coords[min_idx])

        # Estimate reference vessel diameter (d_ref) from neighboring non-stenotic segments
        neighborhood_window = 25
        neigh_start = max(0, min_idx - neighborhood_window)
        neigh_end = min(n_points, min_idx + neighborhood_window + 1)
        neighbor_diameters = np.concatenate([
            smooth_diameters[neigh_start:max(neigh_start, min_idx - 3)],
            smooth_diameters[min(neigh_end, min_idx + 4):neigh_end]
        ])

        if len(neighbor_diameters) > 0:
            d_ref = float(np.percentile(neighbor_diameters, 75))
        else:
            d_ref = float(np.percentile(smooth_diameters[valid_indices], 75))

        d_ref = max(d_ref, d_min + 1e-4)

        # Calculate percentage stenosis
        stenosis_pct = round(float(max(0.0, min(100.0, (1.0 - (d_min / d_ref)) * 100.0))), 1)

        # Lesion severity grading rules:
        # MILD: < 50%
        # MODERATE: 50% <= Stenosis < 70%
        # SEVERE: >= 70% (Catheter Intervention Recommended)
        if stenosis_pct >= 70.0:
            severity = "SEVERE"
            intervention = True
        elif stenosis_pct >= 50.0:
            severity = "MODERATE"
            intervention = False
        else:
            severity = "MILD"
            intervention = False

        centerline_coords = [{"x": int(x_coords[i]), "y": int(y_coords[i])} for i in range(0, len(x_coords), max(1, len(x_coords)//50))]

        return {
            "stenosis_percentage": stenosis_pct,
            "severity_grade": severity,
            "d_min": round(d_min * pixel_spacing_mm, 2),
            "d_ref": round(d_ref * pixel_spacing_mm, 2),
            "lesion_coordinates": {"x": x_min, "y": y_min},
            "intervention_recommended": intervention,
            "centerline_points_count": len(x_coords),
            "centerline_coords": centerline_coords,
        }

    def render_qca_visualization(
        self,
        original_image_input: Any,
        binary_mask: np.ndarray,
        qca_metrics: Dict[str, Any],
        output_save_path: str
    ) -> str:
        """
        Renders an annotated QCA visual diagnostic image with:
          - Clean semi-transparent cyan highlight (#00D2FF) ONLY over the segmented coronary artery
          - Smooth green arterial centerline along vessel path
          - Red circle & crosshair at bottleneck minimum lumen diameter (d_min)
          - High-contrast clinical metric header banner
        """
        if isinstance(original_image_input, (str, Path)) and os.path.exists(original_image_input):
            try:
                _, frames = parse_dicom_or_image(original_image_input)
                bg_img = cv2.cvtColor(frames[0], cv2.COLOR_GRAY2BGR)
            except Exception:
                bg_img = cv2.imread(str(original_image_input), cv2.IMREAD_COLOR)
        elif isinstance(original_image_input, np.ndarray):
            bg_img = original_image_input.copy()
            if bg_img.ndim == 2:
                bg_img = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2BGR)
        else:
            bg_img = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

        h, w = bg_img.shape[:2]
        vis_img = bg_img.copy()

        # 1. Clean Semi-Transparent Cyan Highlight (#00D2FF -> BGR: 255, 210, 0) strictly over segmented vessel
        mask_bool = binary_mask > 0
        if np.any(mask_bool):
            cyan_bgr = np.array([255, 210, 0], dtype=np.uint8)
            alpha = 0.40
            vis_img[mask_bool] = (vis_img[mask_bool] * (1.0 - alpha) + cyan_bgr * alpha).astype(np.uint8)

        # 2. Draw Smooth Green Arterial Centerline
        skeleton, _, y_coords, x_coords = self.extract_centerline_and_diameters(binary_mask)
        if len(x_coords) > 1:
            pts = np.column_stack((x_coords, y_coords)).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_img, [pts], isClosed=False, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        # 3. Highlight Bottleneck (d_min) with Red Crosshair and Marker
        lx = qca_metrics["lesion_coordinates"]["x"]
        ly = qca_metrics["lesion_coordinates"]["y"]
        stenosis = qca_metrics["stenosis_percentage"]
        severity = qca_metrics["severity_grade"]
        d_min = qca_metrics["d_min"]
        d_ref = qca_metrics["d_ref"]
        intervention = qca_metrics["intervention_recommended"]

        radius = max(6, int(d_ref))
        cv2.circle(vis_img, (lx, ly), radius + 8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(vis_img, (lx, ly), (0, 0, 255), cv2.MARKER_CROSS, markerSize=14, thickness=2)


        # 4. Render High-Contrast Top Banner
        banner_height = 50
        banner = np.zeros((banner_height, w, 3), dtype=np.uint8)

        color_map = {
            "SEVERE": (0, 0, 255),      # Red
            "MODERATE": (0, 165, 255),  # Orange
            "MILD": (0, 255, 0)         # Green
        }
        status_color = color_map.get(severity, (255, 255, 255))
        interv_str = "INTERVENTION RECOMMENDED" if intervention else "CONSERVATIVE MANAGEMENT"

        cv2.rectangle(banner, (0, 0), (w, banner_height), (30, 30, 30), -1)
        text_str = f"QCA Stenosis: {stenosis}% ({severity}) | d_min: {d_min}px | d_ref: {d_ref}px | {interv_str}"
        cv2.putText(banner, text_str, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)

        combined_img = np.vstack([banner, vis_img])

        os.makedirs(os.path.dirname(output_save_path), exist_ok=True)
        cv2.imwrite(output_save_path, combined_img)
        return output_save_path


qca_service = QCAService()

