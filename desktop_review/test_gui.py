"""Real Tk/worker integration test using synthetic images and zero models."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import tkinter as tk
import unittest

try:
    from .gui import ReviewApplication, load_base_gui, matching_review_row
except ImportError:
    from gui import ReviewApplication, load_base_gui, matching_review_row


def smoke_test(package_root, research_root, output_root):
    from PIL import Image, ImageDraw, ImageGrab
    out = Path(output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = load_base_gui(package_root)
    photo = out / "synthetic_input.png"
    image = Image.new("RGB", (680, 440), (220, 231, 217))
    draw = ImageDraw.Draw(image)
    for coords in ((90, 120, 235, 230), (400, 210, 535, 310), (610, 45, 623, 58)):
        draw.ellipse(coords, fill=(120, 143, 152))
    image.save(photo)
    photo2 = out / "synthetic_input_2.png"
    image.save(photo2)
    records, engines, originals = [], [], []

    class Switch:
        enabled = False
        def set_enabled(self, value):
            assert threading.current_thread() is threading.main_thread()
            self.enabled = bool(value)
    switch = Switch()

    class Engine(base.SyntheticEngine):
        def __init__(self, root, **kwargs):
            super().__init__(root, **kwargs)
            assert threading.current_thread() is not threading.main_thread()
            self.mode, self.closed = kwargs["mode"], False
            engines.append(self)
        def predict(self, image, detection_mode="normal", **kwargs):
            time.sleep(.08)
            result = super().predict(image, detection_mode)
            result["boxes"].append(deepcopy(result["boxes"][0]))
            result["boxes"][2]["xyxy"] = [606, 40, 629, 63]
            for i, box in enumerate(result["boxes"]):
                box["bird_id"] = f"bird_{i+1:03d}"
                box["species_info"].update(feature_index=10 + i, common_name=f"原种名 {i+1}",
                                          confidence=.98 - i * .1, accepted=i != 2)
            result["synthetic_mode"] = self.mode
            return result
        def close(self):
            self.closed = True

    def saver(image, result, source, folder):
        public = base.save_synthetic(image, result, source, folder)
        originals.append(deepcopy(public["boxes"]))
        enabled = switch.enabled
        states = ["advice", "no_advice", "group_conflict"]
        if not enabled:
            states = ["disabled"] * 3
        elif result["synthetic_mode"] == "baseline":
            states = ["not_supported"] * 3
        rows = [{"bird_id": b["bird_id"], "feature_index": b["species_info"]["feature_index"],
                 "xyxy": b["xyxy"], "original_display_name": f"原种名 {i+1}", "status": states[i],
                 "reason": f"合成状态 {i+1}，不代表模型结论", "group_id": ("willow", "thrush", "conflict")[i],
                 "group_label": ("柳莺合成组", "鸫类合成组", "冲突合成组")[i],
                 "proposal_common_name": "独立建议示例" if states[i] == "advice" else None,
                 "conditional_confidence": .92 if states[i] == "advice" else None, "group_mass": .87}
                for i, b in enumerate(public["boxes"])]
        public["similar_species_review"] = {"status": "synthetic", "message": "合成复核示例", "rows": rows[::-1]}
        records.append((enabled, result["detection_mode"], result["synthetic_mode"]))
        return public

    window = tk.Tk()
    app = ReviewApplication(window, package_root, research_root=research_root, output_root=out,
                            engine_factory=Engine, saver=saver, synthetic=True)
    app.backend = switch  # Only test the main-thread batch snapshot contract.
    assert not app.review_enabled.get() and app.mode.get() == "高精度融合版"
    assert set(base.DETECTION_MODES.values()) == {"normal", "tiled"}
    assert app.log_path.parent == out / "logs"

    def pump_until(predicate, seconds=20):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            window.update()
            if predicate():
                return
            time.sleep(.015)
        raise AssertionError("Synthetic GUI did not finish in time")

    try:
        app.add_paths([photo, photo2])
        app.review_enabled.set(True)
        app.start_batch()
        assert app.batch_review_enabled and "disabled" in app.review_toggle.state()
        app.review_enabled.set(False)  # Programmatic change cannot affect a running batch.
        app.detection_mode.set("普通整图（更快）")
        pump_until(lambda: not app.busy and len(records) == 2)
        assert records == [(True, "tiled", "selected")] * 2
        assert "disabled" not in app.review_toggle.state()
        app.show_job("photo_1")
        for i, needle in enumerate(("独立建议示例", "该组未启用建议", "多个组冲突")):
            app.bird_tree.selection_set(str(i))
            app.select_bird()
            window.update()
            assert needle in app.review_text.get(), app.review_text.get()
            assert f"原种名 {i+1}" in app.crop_caption.get()
            assert app.crop_source is not None
        assert app.current_result["boxes"] == originals[0]
        assert [app.bird_tree.item(str(i), "values") for i in range(3)] == [
            ("1. 原种名 1", "98.0%", "已确认"), ("2. 原种名 2", "88.0%", "已确认"),
            ("3. 原种名 3", "78.0%", "待确认")]
        box = app.current_result["boxes"][0]
        row = app.current_result["similar_species_review"]["rows"][-1]
        assert matching_review_row(box, [row, row]) is None
        assert matching_review_row(box, [{**row, "feature_index": 99}]) is None
        assert matching_review_row(box, [{**row, "xyxy": [0, 0, 1, 1]}]) is None
        app.bird_tree.selection_set("0")
        app.select_bird()
        app.start_batch()
        pump_until(lambda: not app.busy and len(records) == 3)
        assert records[-1] == (False, "normal", "selected") and len(engines) == 1
        assert "本批复核关闭" in app.review_text.get()
        app.mode.set("上一版 N")
        app.review_enabled.set(True)
        app.start_batch()
        pump_until(lambda: not app.busy and len(records) == 4)
        assert records[-1] == (True, "normal", "baseline") and len(engines) == 2
        assert "N 对照不支持" in app.review_text.get() and engines[0].closed
        app.show_review_flow()
        window.update()
        flow_text = app.modal_windows[-1].winfo_children()[0].get("1.0", "end")
        assert "no_advice" in flow_text and "不是没有鸟" in flow_text
        app.modal_windows[-1].destroy()
        layout = []
        for width, height in ((1280, 900), (1100, 760), (960, 650)):
            window.geometry(f"{int(width*app.ui_scale)}x{int(height*app.ui_scale)}")
            window.update()
            bounds = {}
            for name in ("review_toggle", "review_flow_button", "status_label", "run_button"):
                widget = getattr(app, name)
                right = widget.winfo_rootx() - window.winfo_rootx() + widget.winfo_width()
                bottom = widget.winfo_rooty() - window.winfo_rooty() + widget.winfo_height()
                assert right <= window.winfo_width() and bottom <= window.winfo_height(), (name, right, bottom)
                bounds[name] = [right, bottom]
            assert app.preview.winfo_height() >= 80 and app.bird_tree.winfo_height() >= 70, {
                "preview": app.preview.winfo_height(), "birds": app.bird_tree.winfo_height(), "bounds": bounds}
            layout.append({"size": [window.winfo_width(), window.winfo_height()], "bounds": bounds})
        window.geometry(f"{int(1180*app.ui_scale)}x{int(840*app.ui_scale)}")
        app.show_job("photo_2")  # Original selected run, showing independent advice.
        window.update()
        screenshot = out / "review_gui_synthetic.png"
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            ancestor = ctypes.windll.user32.GetAncestor
            ancestor.argtypes, ancestor.restype = [wintypes.HWND, wintypes.UINT], wintypes.HWND
            def capture_window():
                return ImageGrab.grab(window=ancestor(window.winfo_id(), 2))
        else:
            def capture_window():
                return ImageGrab.grab((window.winfo_rootx(), window.winfo_rooty(),
                                       window.winfo_rootx()+window.winfo_width(), window.winfo_rooty()+window.winfo_height()))
        capture_window().save(screenshot)
        app.result_scroll.yview_moveto(1.)
        window.update()
        review_screenshot = out / "review_gui_synthetic_review.png"
        capture_window().save(review_screenshot)
        report = {"passed": True, "synthetic_only": True, "models_loaded": False,
                  "CUDA_run": False, "torch_imported": "torch" in sys.modules, "bird_rows": 3,
                  "multi_bird_identity_and_original_table_preserved": True,
                  "normal_tiled_snapshot": True, "batch_review_snapshot": True,
                  "N_control_not_supported": True, "identity_conflicts_abstain": True,
                  "small_bird_crop_preserved": True, "layout": layout, "screenshot": str(screenshot)}
        report["review_screenshot"] = str(review_screenshot)
        report["screenshot_method"] = "Pillow native window capture" if os.name == "nt" else "desktop region"
        assert not report["torch_imported"]
    finally:
        app.request_close()
        deadline = time.monotonic() + 10
        while app.worker.is_alive() and time.monotonic() < deadline:
            try:
                window.update()
            except tk.TclError:
                break
            time.sleep(.02)
        app.worker.join(2)
        app.close_logger()
        try:
            window.destroy()
        except tk.TclError:
            pass
    report["safe_shutdown"] = not app.worker.is_alive() and all(e.closed for e in engines)
    assert report["safe_shutdown"]
    report["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), Path(__file__).with_name("gui.py"))}
    (out / "gui_smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return report


class DesktopReviewGUITest(unittest.TestCase):
    def test_full_synthetic_gui(self):
        package = os.environ.get("BIRD_NVIDIA_PACKAGE")
        research = os.environ.get("BIRD_RESEARCH_ROOT")
        if not package or not research:
            self.skipTest("Use desktop_review/launch.py --test to configure the installed runtime")
        output = Path(os.environ.get("BIRD_REVIEW_OUTPUT", Path(__file__).parents[1] / "outputs/desktop-review"))
        smoke_test(package, research, output / "gui-smoke")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    smoke_test(args.package_root, args.research_root, args.output_root)
