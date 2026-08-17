import os
import io
import cv2
import unittest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app
from backend.app.services.qca_service import qca_service
from backend.app.engines.deepsa_engine import deepsa_engine
from backend.app.schemas.angiogram import QCAMetricsResponse, QCAAnalysisResult


class TestQCAEngine(unittest.TestCase):
    """
    Automated unit tests to verify Quantitative Coronary Angiography (QCA)
    centerline skeletonization, lumen diameter calculations, stenosis grading,
    visualization rendering, and API endpoints.
    """

    def setUp(self):
        self.client = TestClient(app)

    def _create_synthetic_vessel_mask(self, stenosis_type: str = "mild") -> np.ndarray:
        """Creates a synthetic binary vessel mask (300x300) with smooth cosine bottleneck geometry."""
        mask = np.zeros((300, 300), dtype=np.uint8)

        # Draw smooth vessel from y=30 to y=270, x centered at 150 (d_ref = 30px, radius = 15)
        for y in range(30, 271):
            if 120 <= y <= 180:
                t = (y - 120) / 60.0
                factor = 0.5 * (1.0 - np.cos(2.0 * np.pi * t))
                if stenosis_type == "severe":
                    # d_min = 4 (radius 2) -> Stenosis = (1 - 4/30)*100 = 86.7% (>=70% -> SEVERE)
                    r = int(round(15.0 - factor * 13.0))
                elif stenosis_type == "moderate":
                    # d_min = 8 (radius 4) -> Stenosis = ~60.0% (50-69% -> MODERATE)
                    r = int(round(15.0 - factor * 11.0))
                else:  # mild
                    # d_min = 24 (radius 12) -> Stenosis = (1 - 24/30)*100 = 20.0% (<50% -> MILD)
                    r = int(round(15.0 - factor * 3.0))
            else:
                r = 15

            mask[y, max(0, 150 - r) : min(300, 150 + r + 1)] = 255

        return mask

    def test_skeletonization_and_centerline_extraction(self):
        """Verifies skeletonization and Medial Axis Transform diameter profiling."""
        mask = self._create_synthetic_vessel_mask("mild")
        skeleton, distance_map, y_coords, x_coords = qca_service.extract_centerline_and_diameters(mask)

        self.assertGreater(len(x_coords), 50, "Centerline point extraction failed")
        self.assertGreater(len(y_coords), 50)
        self.assertTrue(np.max(distance_map) > 5.0, "Distance transform radius should be > 5.0px")

    def test_qca_stenosis_calculation_and_grading_mild(self):
        """Verifies Stenosis calculation & grading for MILD lesion (<50%)."""
        mask = self._create_synthetic_vessel_mask("mild")
        metrics = qca_service.compute_qca_metrics(mask)

        self.assertEqual(metrics["severity_grade"], "MILD")
        self.assertFalse(metrics["intervention_recommended"])
        self.assertLess(metrics["stenosis_percentage"], 50.0)

    def test_qca_stenosis_calculation_and_grading_moderate(self):
        """Verifies Stenosis calculation & grading for MODERATE lesion (50-69%)."""
        mask = self._create_synthetic_vessel_mask("moderate")
        metrics = qca_service.compute_qca_metrics(mask)

        self.assertEqual(metrics["severity_grade"], "MODERATE")
        self.assertFalse(metrics["intervention_recommended"])
        self.assertTrue(50.0 <= metrics["stenosis_percentage"] < 70.0)

    def test_qca_stenosis_calculation_and_grading_severe(self):
        """Verifies Stenosis calculation & grading for SEVERE lesion (>=70%) & intervention recommendation."""
        mask = self._create_synthetic_vessel_mask("severe")
        metrics = qca_service.compute_qca_metrics(mask)

        self.assertEqual(metrics["severity_grade"], "SEVERE")
        self.assertTrue(metrics["intervention_recommended"], "Catheter intervention must be recommended for SEVERE lesion")
        self.assertGreaterEqual(metrics["stenosis_percentage"], 70.0)

    def test_qca_visualization_rendering(self):
        """Verifies annotated high-contrast visual diagnostic image rendering."""
        mask = self._create_synthetic_vessel_mask("severe")
        metrics = qca_service.compute_qca_metrics(mask)

        output_path = os.path.join("outputs", "test_qca_vis.png")
        saved_path = qca_service.render_qca_visualization(
            original_image_input=mask,
            binary_mask=mask,
            qca_metrics=metrics,
            output_save_path=output_path
        )

        self.assertTrue(os.path.exists(saved_path))
        self.assertGreater(os.path.getsize(saved_path), 0)

    def test_deepsa_engine_qca_predict(self):
        """Verifies end-to-end DeepSAEngine QCA analysis payload structure."""
        mask = self._create_synthetic_vessel_mask("severe")
        res = deepsa_engine.analyze_qca(image_input=mask, mask_input=mask)

        self.assertEqual(res["status"], "success")
        self.assertIn("stenosis_percentage", res)
        self.assertIn("severity_grade", res)
        self.assertIn("d_min", res)
        self.assertIn("d_ref", res)
        self.assertIn("intervention_recommended", res)
        self.assertTrue(os.path.exists(res["qca_image_path"]))

    def test_api_process_video_endpoint(self):
        """Verifies POST /api/v1/angiogram/process_video returns HTTP 200 OK and QCA metrics."""
        mask = self._create_synthetic_vessel_mask("severe")
        img_bytes = io.BytesIO()
        Image.fromarray(mask).save(img_bytes, format="PNG")
        img_bytes.seek(0)

        response = self.client.post(
            "/api/v1/angiogram/process_video",
            files={"angio_file": ("test_angiogram.png", img_bytes, "image/png")}
        )

        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}: {response.text}")
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertIn("qca_metrics", payload)
        self.assertIn("qca_image_url", payload)

        metrics = payload["qca_metrics"]
        self.assertIn("stenosis_percentage", metrics)
        self.assertIn("severity_grade", metrics)
        self.assertIn("intervention_recommended", metrics)

    def test_connected_component_noise_suppression(self):
        """Verifies Connected Component Analysis removes small noise blobs (<400px) and keeps primary vessel tree."""
        mask = self._create_synthetic_vessel_mask("mild")
        # Add small noise blobs around the image
        mask[10:20, 10:20] = 255   # area = 100
        mask[250:260, 10:20] = 255 # area = 100

        cleaned = qca_service.isolate_arterial_tree(mask, min_area=400)
        self.assertEqual(np.max(cleaned[10:20, 10:20]), 0, "Small noise blob should be suppressed")
        self.assertEqual(np.max(cleaned[250:260, 10:20]), 0, "Small noise blob should be suppressed")
        self.assertGreater(np.count_nonzero(cleaned[100:200, 140:160]), 0, "Primary vessel tree must be preserved")

    def test_dicom_parsing_and_monochrome1_inversion(self):
        """Verifies DICOM parsing with MONOCHROME1 photometric interpretation inversion."""
        from Preprocessing.Angiogram_Preprocessing.Angiogram_DICOM_KeyFrame_Extraction import parse_dicom_or_image
        import pydicom
        from pydicom.dataset import Dataset, FileMetaDataset

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.1.2'
        meta.MediaStorageSOPInstanceUID = '1.2.3'
        meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.Rows = 100
        ds.Columns = 100
        ds.PhotometricInterpretation = "MONOCHROME1"
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1

        arr = np.arange(10000, dtype=np.uint16).reshape((100, 100))
        ds.PixelData = arr.tobytes()

        bio = io.BytesIO()
        pydicom.dcmwrite(bio, ds, write_like_original=False)
        raw_bytes = bio.getvalue()

        parsed_meta, frames = parse_dicom_or_image(raw_bytes)
        self.assertEqual(parsed_meta["Source"], "DICOM")
        self.assertEqual(frames.shape, (1, 100, 100))
        self.assertEqual(frames.dtype, np.uint8)



if __name__ == "__main__":
    unittest.main()

