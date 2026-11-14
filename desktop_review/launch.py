"""Use an existing offline NVIDIA runtime; keep every new write in output_root."""
from pathlib import Path
import argparse
import json
import os
import sys

REPO = Path(__file__).resolve().parents[1]
DLL_HANDLES = []


def find_package():
    configured = os.environ.get("BIRD_NVIDIA_PACKAGE")
    if configured:
        return Path(configured)
    for parent in REPO.parents:
        candidate = parent / "识别模型/v1/v1.1/NVIDIA"
        if (candidate / "app/gui.py").is_file():
            return candidate
    return Path(r"D:\项目\鸟类识别\识别模型\v1\v1.1\NVIDIA")


def configure(package_root, output_root):
    package_root, output_root = Path(package_root).resolve(), Path(output_root).resolve()
    require = ("app/gui.py", "app/engine.py", "configs/recognizer.json", "configs/china_labels.json")
    missing = [name for name in require if not (package_root / name).is_file()]
    if missing:
        raise FileNotFoundError("缺少完整NVIDIA运行包：" + ", ".join(missing)
                                + "；请设置 BIRD_NVIDIA_PACKAGE 或 --package。")
    runtime = package_root / "runtime"
    standard = [p for p in sys.path if p and Path(p).resolve().is_relative_to(runtime)]
    if Path(sys.executable).resolve().is_relative_to(runtime):
        sys.path[:] = [str(REPO), str(package_root / "app")] + [
            str(runtime / "packages" / name) for name in ("primary", "cuda", "base")] + standard
    else:
        sys.path[:0] = [str(REPO), str(package_root / "app")]
    sys.dont_write_bytecode = True
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", YOLO_OFFLINE="true",
                      YOLO_AUTOINSTALL="false", PIP_DISABLE_PIP_VERSION_CHECK="1")
    for name in ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR", "YOLO_CONFIG_DIR", "TORCH_EXTENSIONS_DIR"):
        target = output_root / "cache" / name
        target.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(target)
    for env, suffix in (("TCL_LIBRARY", "tcl8.6"), ("TK_LIBRARY", "tk8.6")):
        if (runtime / "tcl" / suffix).is_dir():
            os.environ[env] = str(runtime / "tcl" / suffix)
    if os.name == "nt":
        for directory in (runtime, runtime / "DLLs", runtime / "packages/cuda/torch/lib"):
            if directory.is_dir():
                DLL_HANDLES.append(os.add_dll_directory(str(directory)))


def main():
    parser = argparse.ArgumentParser(description="逐鸟识别与独立相似种复核GUI")
    parser.add_argument("--package", type=Path, default=find_package())
    parser.add_argument("--research", type=Path,
                        default=Path(os.environ.get("BIRD_RESEARCH_ROOT", str(REPO.parent))))
    parser.add_argument("--output", type=Path, default=REPO / "outputs/desktop-review")
    parser.add_argument("--test", action="store_true", help="CPU路由/后端/真实Tk合成界面测试，不加载视觉模型")
    parser.add_argument("--verify-runtime", action="store_true", help="对已知图片实推，核对普通/分块原结果")
    parser.add_argument("--inspect-saved-gui", action="store_true", help="仅显示并检查已保存GPU结果，不启动推理")
    args = parser.parse_args()
    configure(args.package, args.output)
    os.environ["BIRD_NVIDIA_PACKAGE"] = str(args.package.resolve())
    os.environ["BIRD_RESEARCH_ROOT"] = str(args.research.resolve())
    os.environ["BIRD_REVIEW_OUTPUT"] = str(args.output.resolve())
    if args.test:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromNames([
            "desktop_review.test_routing", "desktop_review.test_backend", "desktop_review.test_gui"])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    if args.verify_runtime:
        from desktop_review.verify_runtime import verify
        result = verify(args.package, args.research, args.output)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result["passed"] else 1
    if args.inspect_saved_gui:
        from desktop_review.verify_runtime import inspect_saved_gui
        result = inspect_saved_gui(args.package, args.research, args.output)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    import tkinter as tk
    from desktop_review.gui import ReviewApplication
    root = tk.Tk()
    ReviewApplication(root, args.package, research_root=args.research, output_root=args.output)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
