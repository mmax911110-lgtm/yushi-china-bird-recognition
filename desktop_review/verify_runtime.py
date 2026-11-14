"""Known-image integration regression, not a new accuracy evaluation."""
from pathlib import Path
from copy import deepcopy
import gc
import json
import time

from PIL import Image, ImageOps

from desktop_review.backend import ReviewBackend, read, sha


def fingerprint(public):
    return [{key: deepcopy(box[key]) for key in
             ("bird_id", "xyxy", "confidence", "species_info", "crop_sha256")}
            for box in public["boxes"]]


def success_conditions(report):
    expected = "0745aa534bd75c331b3977f099bcd60c9637f7d9b4ae35ac2b47608a487fc6c6"
    completed = {"agrees", "suggestion", "uncertain", "outgroup"}
    return (len(report["cases"]) == 4 and len(report["reference_calls"]) == 4
            and all(c["exact_primary_fingerprint_equal"] and c["additional_visual_forwards"] == 0
                    and "unavailable" not in c["review_statuses"] for c in report["cases"])
            and any(c["review_head_sha256"] == expected and completed.intersection(c["review_statuses"])
                    for c in report["cases"])
            and report.get("protected_assets_unchanged") and report.get("source_images_unchanged"))


def inspect_saved_gui(package, research, output):
    """Render an actual saved GPU event, with no inference job or Torch import."""
    import ctypes
    from ctypes import wintypes
    import sys
    import tkinter as tk
    from PIL import ImageGrab
    from desktop_review.gui import ReviewApplication
    output = Path(output)
    report_path = output / "runtime-verification/report.json"
    report = read(report_path)
    require_valid = success_conditions(report)
    if not require_valid:
        raise ValueError("实推验收未通过，不能作为成功界面预览")
    event = Path(next(c["event_dir"] for c in report["cases"] if "uncertain" in c["review_statuses"]))
    public, review = read(event / "result.json"), read(event / "similar_species_review.json")
    public["similar_species_review"] = review
    protected = {str(p): sha(p) for p in (event / "result.json", event / "similar_species_review.json")}
    def guard(*args, **kwargs):
        raise AssertionError("Saved-event preview must not start a model")
    window = tk.Tk()
    app = ReviewApplication(window, package, research_root=research, output_root=output,
                            engine_factory=guard, saver=guard)
    preview = {}
    try:
        window.geometry("1180x840")
        app.add_paths([public["source"]])
        key = next(iter(app.jobs))
        app.jobs[key].update(state="已完成", result=public)
        app.photo_tree.set(key, "state", "已完成")
        app.photo_tree.set(key, "birds", str(len(public["boxes"])))
        app.review_enabled.set(True)  # Display saved batch, never run a job.
        app.subtitle.set("已保存的 GPU 实推结果预览 · 本次不启动推理")
        app.detection_mode.set("普通整图（更快）" if public.get("detection_mode") == "normal" else "分块补检（优先找小鸟）")
        app.show_job(key)
        app.status.set("已载入保存结果；本次没有启动新的识别。")
        window.update()
        app.result_scroll.yview_moveto(1.)
        window.update()
        ancestor = ctypes.windll.user32.GetAncestor
        ancestor.argtypes, ancestor.restype = [wintypes.HWND, wintypes.UINT], wintypes.HWND
        image = ImageGrab.grab(window=ancestor(window.winfo_id(), 2))
        path = output / "gui-real-saved-preview.png"
        image.save(path)
        text = app.review_text.get()
        if "建议证据不足" not in text or app.worker.engine is not None or not app.commands.empty():
            raise AssertionError("GUI did not show the bound saved review or unexpectedly queued inference")
        preview.update(passed=True, actual_saved_GPU_event=True, new_inference_started=False,
                       torch_imported="torch" in sys.modules, event_dir=str(event),
                       original_row_count=len(public["boxes"]), review_text=text, screenshot=str(path))
    finally:
        app.request_close()
        deadline = time.monotonic() + 5
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
    preview["worker_closed"] = not app.worker.is_alive()
    preview["inputs_unchanged"] = all(sha(p) == value for p, value in protected.items())
    if not preview["worker_closed"] or not preview["inputs_unchanged"] or preview["torch_imported"]:
        raise AssertionError("Saved-event display was not isolated")
    (output / "gui-real-saved-preview.json").write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")
    return preview


def verify(package, research, output):
    from engine import BirdRecognizer, save_result
    package, research, output = Path(package), Path(research), Path(output)
    folder = output / "runtime-verification"
    folder.mkdir(parents=True, exist_ok=True)
    previous = read(research / "similar_species_review_20261114/runtime_checks/runtime_verification.json")
    sparrow = Path(previous["reference_calls"][0]["source_path"])
    manifest = read(research / "species_rejection_20261114/manifest.json")
    willow_row = next(row for row in manifest["rows"] if row.get("role") == "val" and
                      row.get("global_id") == 70 and row.get("valid_bird_crop") and Path(row["source_path"]).is_file())
    sources = [sparrow, Path(willow_row["source_path"])]
    protected = read(research / "similar_species_review_20261114/protected_final.json")["files"]
    expected = {row["path"]: row["sha256"] for row in protected}
    original_ok = all(sha(path) == value for path, value in expected.items())
    if not original_ok:
        raise ValueError("原资产保护记录不一致，停止实推")
    source_hashes = {str(path): sha(path) for path in sources}
    report = {"passed": False, "known_validation_images": True, "new_accuracy_test": False,
              "training_run": False, "real_GPU": True, "cases": [], "reference_calls": [],
              "source_sha256": source_hashes, "protected_files_checked": len(expected)}
    engine = None
    try:
        references = {}
        engine = BirdRecognizer(package, device="cuda", mode="selected", progress=lambda msg: print(msg, flush=True))
        for source in sources:
            for mode in ("normal", "tiled"):
                with Image.open(source) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                result = engine.predict(image, detection_mode=mode)
                public = save_result(image, result, source, folder / "original")
                references[(str(source), mode)] = fingerprint(public)
                report["reference_calls"].append({"source": str(source), "mode": mode,
                    "event_dir": public["event_dir"], "boxes": len(public["boxes"])})
                print("reference", source.parent.name, mode, len(public["boxes"]), flush=True)
        engine.close()
        engine = None
        gc.collect()
        adapter = ReviewBackend(package, research, output)
        adapter.set_enabled(True)
        engine = adapter.factory(package, device="cuda", mode="selected", progress=lambda msg: print(msg, flush=True))
        for source in sources:
            for mode in ("normal", "tiled"):
                with Image.open(source) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                started = time.perf_counter()
                result = engine.predict(image, detection_mode=mode)
                public = adapter.save(image, result, source, folder / "integrated")
                same = fingerprint(public) == references[(str(source), mode)]
                review = public["similar_species_review"]
                row = {"source": str(source), "mode": mode, "boxes": len(public["boxes"]),
                       "event_dir": public["event_dir"], "exact_primary_fingerprint_equal": same,
                       "additional_visual_forwards": review["additional_visual_forwards"],
                       "review_statuses": [r["status"] for r in review["rows"]],
                       "review_head_sha256": getattr(adapter.head, "binding", {}).get("head_sha256"),
                       "elapsed_ms": (time.perf_counter() - started) * 1000}
                report["cases"].append(row)
                print("integrated", source.parent.name, mode, "same=" + str(same), row["review_statuses"], flush=True)
                if not same:
                    raise AssertionError("原鸟框、Top5、确认状态或裁图出现变化")
        report["protected_assets_unchanged"] = all(sha(path) == value for path, value in expected.items())
        report["source_images_unchanged"] = all(sha(path) == value for path, value in source_hashes.items())
        report["passed"] = bool(success_conditions(report))
    except Exception as exc:
        report["error"] = repr(exc)
    finally:
        if engine is not None:
            engine.close()
        report["engine_closed"] = True
        path = folder / "report.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
