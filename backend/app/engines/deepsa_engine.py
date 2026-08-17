import os
import sys
import time
import socket
import threading
import subprocess
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from backend.app.core.config import settings
from backend.app.engines.base_engine import BaseMLEngine
from backend.app.services.qca_service import qca_service


class DeepSAEngine(BaseMLEngine):
    """
    DeepSA & QCA Engine for Quantitative Coronary Angiography (QCA).
    Integrates vessel segmentation with automated centerline & diameter profiling,
    percentage stenosis calculation, and lesion severity classification.
    """

    def __init__(self):
        self._proc = None
        self._lock = threading.Lock()
        self._detector = None

    def load_models(self) -> None:
        """Lazily loads StenosisDetector model checkpoint if available."""
        if self._detector is None:
            try:
                from process_02_cardiac_analysis.angiogram_blockage_detection.inference.detector_pipeline import StenosisDetector
                ckpt_path = settings.PROJECT_ROOT / "process_02_cardiac_analysis" / "angiogram_blockage_detection" / "ckpt" / "fscad_36249.ckpt"
                if ckpt_path.exists():
                    self._detector = StenosisDetector(ckpt_path=ckpt_path)
                    logging.info("DeepSA StenosisDetector model loaded successfully.")
            except Exception as e:
                logging.warning(f"Notice: Could not pre-load StenosisDetector ({e}). Fallback segmentation will be used.")

    def is_running(self) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", settings.DEEPSA_PORT), timeout=1):
                return True
        except OSError:
            return False

    def ensure_running(self) -> None:
        with self._lock:
            if self.is_running():
                return
            if not hasattr(settings, 'DEEPSA_SCRIPT') or not Path(settings.DEEPSA_SCRIPT).exists():
                return
            logging.info(f"Starting DeepSA process from: {settings.DEEPSA_SCRIPT}")
            try:
                self._proc = subprocess.Popen(
                    [sys.executable, str(settings.DEEPSA_SCRIPT)],
                    cwd=str(settings.PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                deadline = time.time() + 10
                while time.time() < deadline:
                    if self.is_running():
                        return
                    time.sleep(0.5)
            except Exception as e:
                logging.warning(f"DeepSA service start skipped: {e}")

    def generate_vessel_mask(self, image_input: Any) -> np.ndarray:
        """
        Generates binary vessel segmentation mask using DeepSA StenosisDetector,
        or fallback adaptive thresholding / vesselness filter.
        """
        if isinstance(image_input, (str, Path)):
            try:
                from Preprocessing.Angiogram_Preprocessing.Angiogram_DICOM_KeyFrame_Extraction import parse_dicom_or_image
                _, frames = parse_dicom_or_image(image_input)
                img = frames[0]
            except Exception:
                img = cv2.imread(str(image_input), cv2.IMREAD_GRAYSCALE)
        elif isinstance(image_input, np.ndarray):
            img = image_input if image_input.ndim == 2 else cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            raise TypeError("Unsupported image_input for mask generation.")

        # Option A: DeepSA detector model pass if loaded
        if self._detector is not None:
            try:
                debug_res = self._detector.detect(img, return_debug=True)
                seg_img = debug_res.get("seg_img")
                if seg_img is not None:
                    return qca_service.preprocess_mask(seg_img)
            except Exception as de:
                logging.warning(f"DeepSA detector execution warning ({de}). Using adaptive vessel filter.")

        # Option B: Fallback adaptive vessel segmentation filter
        return qca_service.preprocess_mask(img)


    def analyze_qca(
        self,
        image_input: Any,
        mask_input: Optional[Any] = None,
        save_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Performs full Quantitative Coronary Angiography (QCA) analysis on an angiogram frame:
          1. Obtains or extracts segmented vessel binary mask.
          2. Computes QCA metrics (d_min, d_ref, Stenosis %, Severity Grade, Intervention Recommendation).
          3. Renders annotated high-contrast visual diagnostic report.
        """
        if mask_input is not None:
            binary_mask = qca_service.preprocess_mask(mask_input)
        else:
            binary_mask = self.generate_vessel_mask(image_input)

        qca_metrics = qca_service.compute_qca_metrics(binary_mask)

        if save_dir is None:
            save_dir = settings.PROJECT_ROOT / "outputs" / "qca_reports"
        save_dir.mkdir(parents=True, exist_ok=True)

        qca_filename = f"qca_annotated_{int(time.time()*1000)}.png"
        output_vis_path = save_dir / qca_filename

        qca_service.render_qca_visualization(
            original_image_input=image_input,
            binary_mask=binary_mask,
            qca_metrics=qca_metrics,
            output_save_path=str(output_vis_path)
        )

        return {
            "status": "success",
            "stenosis_percentage": qca_metrics["stenosis_percentage"],
            "severity_grade": qca_metrics["severity_grade"],
            "d_min": qca_metrics["d_min"],
            "d_ref": qca_metrics["d_ref"],
            "lesion_coordinates": qca_metrics["lesion_coordinates"],
            "intervention_recommended": qca_metrics["intervention_recommended"],
            "qca_image_path": str(output_vis_path),
            "qca_metrics": qca_metrics,
            "url": getattr(settings, 'DEEPSA_URL', 'http://127.0.0.1:5000'),
            "error": None
        }

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Engine predict interface.
        Accepts:
            data = {"image_path": str} or {"frame_bytes": bytes} or {"mask_path": str}
        """
        image_path = data.get("image_path")
        mask_path = data.get("mask_path")
        save_dir_str = data.get("save_dir")

        save_dir = Path(save_dir_str) if save_dir_str else None

        if image_path and os.path.exists(image_path):
            return self.analyze_qca(image_input=image_path, mask_input=mask_path, save_dir=save_dir)

        if mask_path and os.path.exists(mask_path):
            return self.analyze_qca(image_input=mask_path, mask_input=mask_path, save_dir=save_dir)

        # Fallback dummy mask analysis for generic ping
        dummy_mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.line(dummy_mask, (50, 150), (250, 150), 255, 12)
        cv2.line(dummy_mask, (140, 150), (160, 150), 255, 4)
        return self.analyze_qca(image_input=dummy_mask, mask_input=dummy_mask, save_dir=save_dir)


deepsa_engine = DeepSAEngine()
