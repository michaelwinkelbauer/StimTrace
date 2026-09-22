from __future__ import annotations

import json
import shutil
import unittest
import uuid
import zipfile
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication

from model_training import ModelTrainingPage, create_training_archive, phase_aware_frame_indices


class ModelTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.root = Path.cwd() / f".test-training-archive-{uuid.uuid4().hex}"
        for name in ("frames", "annotations", "masks"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, self.root, True)

    def add_pair(self, stem: str) -> None:
        Image.new("RGB", (8, 8), "white").save(self.root / "frames" / f"{stem}.png")
        mask = Image.new("L", (8, 8), 0)
        mask.putpixel((4, 4), 255)
        mask.save(self.root / "masks" / f"{stem}.png")
        (self.root / "annotations" / f"{stem}.json").write_text(
            json.dumps({"shapes": [{"label": "Pillar"}]}), encoding="utf-8"
        )

    def add_empty_pair(self, stem: str) -> None:
        """Save an explicit empty mask to represent a reviewed negative-control frame."""
        Image.new("RGB", (8, 8), "white").save(self.root / "frames" / f"{stem}.png")
        Image.new("L", (8, 8), 0).save(self.root / "masks" / f"{stem}.png")
        (self.root / "annotations" / f"{stem}.json").write_text(
            json.dumps({"shapes": []}), encoding="utf-8"
        )

    def test_training_archive_requires_two_saved_annotations(self):
        self.add_pair("frame_1")
        with self.assertRaisesRegex(ValueError, "at least two"):
            create_training_archive(self.root)

    def test_compact_training_controls_do_not_overlap(self):
        page = ModelTrainingPage()
        page.resize(640, 480)
        page.show()
        self.app.processEvents()
        controls = page.training_controls
        frame_bottom = page.frame_list.mapTo(controls, QPoint(0, 0)).y() + page.frame_list.height()
        annotation_top = page.class_name.parentWidget().mapTo(controls, QPoint(0, 0)).y()
        training_top = page.model_name.parentWidget().mapTo(controls, QPoint(0, 0)).y()
        self.assertLessEqual(frame_bottom, annotation_top)
        self.assertLess(annotation_top, training_top)
        self.assertGreater(page.controls_scroll.verticalScrollBar().maximum(), 0)
        page.close()

    def test_auto_mask_is_prominent_and_enabled_by_default(self):
        page = ModelTrainingPage()
        self.assertTrue(page.auto_mask_checkbox.isChecked())
        self.assertEqual(page.auto_mask_checkbox.objectName(), "autoMaskNextFrame")
        self.assertIn("recommended", page.auto_mask_checkbox.text().lower())
        self.assertGreaterEqual(page.auto_mask_checkbox.minimumHeight(), 40)
        page.close()

    def test_phase_aware_candidates_are_distinct_and_include_peak_state(self):
        motion = [0, 0.1, 0.8, 0.2, 0.1, 0.7, 0.1, 0]
        sharpness = [4, 5, 3, 6, 6, 3, 5, 4]
        deformation = [0, 0.1, 0.5, 1.0, 0.8, 0.4, 0.1, 0]
        candidates = phase_aware_frame_indices(motion, sharpness, deformation, 10.0, 5)
        indices = [int(candidate["frame_index"]) for candidate in candidates]
        self.assertEqual(len(indices), len(set(indices)))
        self.assertLessEqual(len(indices), 5)
        self.assertIn("maximum deformation", {str(candidate["phase"]) for candidate in candidates})

    def test_phase_aware_archive_writes_split_manifest(self):
        frame_metadata = {}
        for index in range(25):
            stem = f"frame_{index}"
            self.add_pair(stem)
            split = "training" if index < 20 else "validation"
            frame_metadata[f"{stem}.png"] = {
                "source_video_id": "training_video" if split == "training" else "validation_video",
                "source_video": "training.avi" if split == "training" else "validation.avi",
                "split": split,
                "phase": "maximum deformation",
            }
        (self.root / "project.json").write_text(json.dumps({
            "selection_strategy": "phase_aware_v1",
            "effort": "minimum",
            "frame_metadata": frame_metadata,
        }), encoding="utf-8")
        archive, count = create_training_archive(self.root)
        self.assertEqual(count, 25)
        with zipfile.ZipFile(archive) as bundle:
            manifest = json.loads(bundle.read("dataset_manifest.json"))
        self.assertEqual(manifest["summary"]["training_frames"], 20)
        self.assertEqual(manifest["summary"]["validation_frames"], 5)

    def test_training_archive_includes_saved_empty_negative_controls(self):
        frame_metadata = {}
        for index in range(25):
            stem = f"frame_{index}"
            (self.add_empty_pair if index == 0 else self.add_pair)(stem)
            split = "training" if index < 20 else "validation"
            frame_metadata[f"{stem}.png"] = {
                "source_video_id": "training_video" if split == "training" else "validation_video",
                "source_video": "training.avi" if split == "training" else "validation.avi",
                "split": split,
                "phase": "maximum deformation",
            }
        (self.root / "project.json").write_text(json.dumps({
            "selection_strategy": "phase_aware_v1",
            "effort": "minimum",
            "frame_metadata": frame_metadata,
        }), encoding="utf-8")

        archive, count = create_training_archive(self.root)
        self.assertEqual(count, 25)
        with zipfile.ZipFile(archive) as bundle:
            manifest = json.loads(bundle.read("dataset_manifest.json"))
        self.assertEqual(manifest["summary"]["negative_control_frames"], 1)
        self.assertEqual(manifest["summary"]["positive_mask_frames"], 24)

    def test_legacy_project_is_rejected(self):
        self.add_pair("frame_1")
        self.add_pair("frame_2")
        with self.assertRaisesRegex(ValueError, "older annotation project"):
            create_training_archive(self.root)


if __name__ == "__main__":
    unittest.main()
