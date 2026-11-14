"""CPU tests for crop identity, frozen math and immutable primary output."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from desktop_review import backend

PACKAGE = Path(os.environ.get("BIRD_NVIDIA_PACKAGE", r"D:\项目\鸟类识别\识别模型\v1\v1.1\NVIDIA"))
RESEARCH = Path(os.environ.get("BIRD_RESEARCH_ROOT", str(Path(__file__).resolve().parents[2])))


class BackendTests(unittest.TestCase):
    def setUp(self):
        if not (PACKAGE / "configs/china_labels.json").is_file():
            self.skipTest("需要外部NVIDIA类别资产；不是可独立运行的完整模型包")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.app = backend.ReviewBackend(PACKAGE, RESEARCH, self.temp.name)
        self.app.engine = SimpleNamespace(mode="selected")
        from inference.engine_n import square_context
        self.square = square_context
        self.image = Image.new("RGB", (180, 100), (120, 150, 100))

    def make_result(self, predictions):
        boxes, crops, features = [], [], {}
        for i, (a, b) in enumerate(predictions):
            xyxy = [5 + i * 35, 10, 30 + i * 35, 40]
            info = {"name": self.app.labels[a]["scientific_name"], "display_name": self.app.labels[a]["common_name"],
                    "accepted": i % 2 == 0, "confidence": .55, "feature_index": i,
                    "top5": [{"name": self.app.labels[j]["scientific_name"], "confidence": p}
                             for j, p in ((a, .55), (b, .45))]}
            boxes.append({"xyxy": xyxy, "confidence": .6, "species_info": info})
            crops.append(self.square(self.image, xyxy))
            features[i] = np.ones(1024, np.float32) * (i + 1)
        return {"boxes": boxes, "_crops": crops, "_review_bio25": features}

    def public_box(self, result, i=0):
        return {"bird_id": f"bird_{i + 1:03d}", "crop_sha256": backend.image_sha(result["_crops"][i])}

    def fake_head(self, candidate=70, outgroup=False):
        return SimpleNamespace(
            targets=[70, 126, 198, 473],
            rule={"trigger_max_margin": .5, "trigger_min_pair_mass": .3, "expert_min_confidence": .85,
                  "expert_min_margin": .25, "expert_min_domain_score": 0.},
            binding={"synthetic_only": True},
            predict=lambda feature: {"candidate_global_id": candidate, "rejected_as_OTHER": outgroup,
                                     "conditional_confidence": .98, "conditional_margin": .96,
                                     "group_mass": .99})

    def test_each_crop_has_own_status_and_coordinates(self):
        self.app.set_enabled(True)
        result = self.make_result([(70, 126), (1, 88), (3, 50)])
        original = deepcopy(result["boxes"])
        self.app.head = self.fake_head()
        rows = [self.app._review_row(self.image, result, self.public_box(result, i), i) for i in range(3)]
        self.assertEqual([r["status"] for r in rows], ["agrees", "not_enabled", "not_triggered"])
        self.assertEqual([r["feature_index"] for r in rows], [0, 1, 2])
        self.assertEqual([r["xyxy"] for r in rows], [r["xyxy"] for r in original])
        self.assertEqual(result["boxes"], original)

    def test_disagreement_is_only_advice_even_on_unknown(self):
        self.app.set_enabled(True)
        self.app.head = self.fake_head(126)
        result = self.make_result([(70, 126)])
        result["boxes"][0]["species_info"]["accepted"] = False
        original = deepcopy(result["boxes"])
        row = self.app._review_row(self.image, result, self.public_box(result), 0)
        self.assertEqual(row["status"], "suggestion")
        self.assertEqual(row["proposal_global_id"], 126)
        self.assertFalse(row["original_accepted"] or row["applied_to_original"])
        self.assertEqual(result["boxes"], original)

    def test_OTHER_does_not_remove_a_bird(self):
        self.app.set_enabled(True)
        self.app.head = self.fake_head(outgroup=True)
        result = self.make_result([(70, 126)])
        row = self.app._review_row(self.image, result, self.public_box(result), 0)
        self.assertEqual(row["status"], "outgroup")
        self.assertEqual(len(result["boxes"]), 1)
        self.assertEqual(row["proposal_global_id"], -1)

    def test_confident_main_is_reviewed_but_original_advice_gate_stays(self):
        self.app.set_enabled(True)
        called = []
        self.app.head = self.fake_head()
        predict = self.app.head.predict
        self.app.head.predict = lambda feature: (called.append(True), predict(feature))[1]
        result = self.make_result([(70, 126)])
        result["boxes"][0]["species_info"]["top5"][0]["confidence"] = .99
        result["boxes"][0]["species_info"]["top5"][1]["confidence"] = .01
        row = self.app._review_row(self.image, result, self.public_box(result), 0)
        self.assertTrue(called)
        self.assertEqual(row["status"], "uncertain")

    def test_wrong_crop_fails_closed(self):
        self.app.set_enabled(True)
        self.app.head = self.fake_head()
        result = self.make_result([(70, 126)])
        public = self.public_box(result)
        public["crop_sha256"] = "0" * 64
        original = deepcopy(result["boxes"])
        row = self.app._review_row(self.image, result, public, 0)
        self.assertEqual(row["status"], "unavailable")
        self.assertEqual(result["boxes"], original)

    def test_baseline_N_remains_usable(self):
        self.app.set_enabled(True)
        self.app.engine.mode = "baseline"
        result = self.make_result([(70, 126)])
        row = self.app._review_row(self.image, result, self.public_box(result), 0)
        self.assertEqual(row["status"], "not_supported")

    def test_missing_specialist_preserves_primary(self):
        self.app.set_enabled(True)
        self.app.research_root = Path(self.temp.name) / "absent"
        result = self.make_result([(70, 126)])
        original = deepcopy(result["boxes"])
        row = self.app._review_row(self.image, result, self.public_box(result), 0)
        self.assertEqual(row["status"], "unavailable")
        self.assertEqual(result["boxes"], original)

    def test_save_keeps_original_json_and_separates_sidecar(self):
        self.app.set_enabled(True)
        self.app.head = self.fake_head(126)
        result = self.make_result([(70, 126), (1, 88)])
        original = deepcopy(result["boxes"])
        public = self.app.save(self.image, result, Path(self.temp.name) / "synthetic.png", self.temp.name)
        saved = backend.read(public["result_json"])
        self.assertNotIn("similar_species_review", saved)
        self.assertEqual(backend.sha(public["result_json"]), public["similar_species_review"]["original_result_sha256"])
        self.assertEqual(backend.read(public["review_json"])["rows"][0]["status"], "suggestion")
        self.assertEqual(result["boxes"], original)
        self.assertTrue((Path(public["event_dir"]) / "similar_species_review.html").is_file())

    def test_frozen_head_matches_original_inference_math(self):
        if not (RESEARCH / "species_rejection_20261114/frozen_expert.json").is_file():
            self.skipTest("未提供原柳莺冻结实验资产")
        head = backend.WillowHead(PACKAGE, RESEARCH)
        source = RESEARCH / "species_rejection_20261114/train_rejection.py"
        spec = importlib.util.spec_from_file_location("review_reference_rejection", source)
        reference = importlib.util.module_from_spec(spec)
        old_path = sys.path.copy()
        try:
            spec.loader.exec_module(reference)
            previous = reference.load_checkpoint(source.parent / "heads/species_aware_OTHER_bio25_C0p1/head.npz")
            rng = np.random.default_rng(125)
            features = rng.normal(size=(7, 1024)).astype(np.float32)
            expected = previous.predict({"bio25": features}, threshold=0.)
            for i, feature in enumerate(features):
                actual = head.predict(feature)
                np.testing.assert_allclose(actual["conditional_probabilities"], expected["q"][i], rtol=1e-12, atol=1e-13)
                self.assertEqual(actual["rejected_as_OTHER"], bool(expected["reject_decision"][i]))
                self.assertAlmostEqual(actual["group_mass"], expected["domain"][i], places=12)
        finally:
            sys.path[:] = old_path
        self.assertNotIn("torch", sys.modules)

    def test_sidecar_write_failure_keeps_successful_original(self):
        self.app.set_enabled(True)
        self.app.head = self.fake_head()
        result = self.make_result([(70, 126)])
        original_write = Path.write_text
        def fail_only_optional(path, *args, **kwargs):
            if path.name.startswith("similar_species_review"):
                raise PermissionError("synthetic optional sidecar failure")
            return original_write(path, *args, **kwargs)
        with patch.object(Path, "write_text", fail_only_optional):
            public = self.app.save(self.image, result, Path(self.temp.name) / "synthetic.png", self.temp.name)
        self.assertTrue(Path(public["result_json"]).is_file())
        self.assertIn("export_error", public["similar_species_review"])
        self.assertEqual(len(public["boxes"]), 1)

    def test_row_exception_is_isolated_from_original_success(self):
        result = self.make_result([(70, 126)])
        with patch.object(self.app, "_review_row", side_effect=RuntimeError("synthetic row failure")):
            public = self.app.save(self.image, result, Path(self.temp.name) / "synthetic.png", self.temp.name)
        self.assertTrue(Path(public["result_json"]).is_file())
        self.assertEqual(public["similar_species_review"]["rows"][0]["status"], "unavailable")

    def test_different_main_config_cannot_reuse_specialist_space(self):
        other = Path(self.temp.name) / "other-package/configs"
        other.mkdir(parents=True)
        (other / "china_labels.json").write_bytes((PACKAGE / "configs/china_labels.json").read_bytes())
        config = backend.read(PACKAGE / "configs/recognizer.json")
        config["synthetic_changed_config"] = True
        (other / "recognizer.json").write_text(__import__("json").dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "特征空间"):
            backend.WillowHead(other.parent, RESEARCH)


class CaptureTests(unittest.TestCase):
    def test_tile_feature_offsets_and_method_restoration(self):
        class Encoder:
            def encode_images(self, crops):
                return np.asarray([[c.getpixel((0, 0))[0]] * 1024 for c in crops], np.float32), 0.
        class Parent:
            def _apply_selected(self, original, result):
                if original["boxes"]:
                    self.p_encoder.encode_images([original["_crops"][b["species_info"]["feature_index"]]
                                                 for b in original["boxes"]])
                return True, 0., 0.
            def predict(self, image, **kwargs):
                result = {"boxes": [], "_crops": []}
                for count, fi, value in [(6, 4, 11), (3, 1, 22), (0, None, 0)]:
                    block = {"boxes": [] if fi is None else [{"species_info": {"feature_index": fi}}],
                             "_crops": [Image.new("RGB", (2, 2), (value, 0, 0)) for _ in range(count)]}
                    self._apply_selected(block, {})
                    offset = len(result["_crops"])
                    result["_crops"].extend(block["_crops"])
                    if fi is not None:
                        result["boxes"].append({"species_info": {"feature_index": offset + fi}})
                return result
        class Fake(backend.CaptureMixin, Parent):
            mode = "selected"
            def __init__(self):
                self.p_encoder = Encoder()
        fake = Fake()
        result = fake.predict(Image.new("RGB", (2, 2)), detection_mode="tiled")
        self.assertEqual(sorted(result["_review_bio25"]), [4, 7])
        self.assertEqual(result["_review_bio25"][4][0], 11)
        self.assertEqual(result["_review_bio25"][7][0], 22)
        self.assertNotIn("encode_images", fake.p_encoder.__dict__)


if __name__ == "__main__":
    unittest.main()
