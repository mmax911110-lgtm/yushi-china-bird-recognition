"""Full photo application with an independent, default-off review panel."""
from __future__ import annotations

import importlib.util
import logging
import sys
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk


def load_base_gui(package_root):
    path = (Path(package_root) / "app" / "gui.py").resolve()
    cached = sys.modules.get("bird_base_gui")
    if cached is not None and Path(cached.__file__).resolve() == path:
        return cached
    spec = importlib.util.spec_from_file_location("bird_base_gui", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def matching_review_row(box, rows):
    """Require a unique, consistent identity; never bind by list position."""
    fi = box.get("species_info", {}).get("feature_index", box.get("feature_index"))
    bid = box.get("bird_id")
    matches = [r for r in rows if fi is not None and r.get("feature_index") == fi]
    if not matches:
        matches = [r for r in rows if bid and r.get("bird_id") == bid
                   and (fi is None or r.get("feature_index") is None)]
    if len(matches) != 1:
        return None
    row = matches[0]
    if bid and row.get("bird_id") and bid != row["bird_id"]:
        return None
    if row.get("xyxy") is not None and list(row["xyxy"]) != list(box.get("xyxy", [])):
        return None
    return row


class ReviewApplication:
    """Bind the shipped BirdApplication at runtime, retaining its complete UI."""
    def __new__(cls, window, package_root, **kwargs):
        if cls is ReviewApplication:
            base = load_base_gui(package_root)
            bound = type("BoundReviewApplication", (cls, base.BirdApplication), {})
            return object.__new__(bound)
        return object.__new__(cls)

    def __init__(self, window, package_root, *, research_root, output_root,
                 engine_factory=None, saver=None, synthetic=False):
        self.research_root = Path(research_root).resolve()
        self.review_enabled = tk.BooleanVar(master=window, value=False)
        self.review_text = tk.StringVar(master=window, value="复核默认关闭；原识别结果保留。")
        self.batch_review_enabled = False
        self.backend = None
        if engine_factory is None or saver is None:
            if __package__:
                from .backend import ReviewBackend
            else:
                from backend import ReviewBackend
            self.backend = ReviewBackend(package_root, self.research_root, output_root)
            engine_factory = engine_factory or self.backend.factory
            saver = saver or self.backend.save
        super().__init__(window, package_root, engine_factory=engine_factory, saver=saver,
                         output_root=output_root, synthetic=synthetic)
        window.title(window.title() + " · 独立相似鸟复核")

    def make_logger(self):
        folder = self.output_root / "logs"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / ("review_gui_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".log")
        logger = logging.getLogger("bird_review." + str(id(self)))
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
        return path, logger

    def build_ui(self):
        super().build_ui()
        settings = self.mode_combo.master
        for widget in settings.winfo_children():
            if "textvariable" in widget.keys() and str(widget.cget("textvariable")) == str(self.device_text):
                widget.grid_configure(columnspan=2)
                widget.configure(wraplength=260)
        tools = ttk.Frame(settings)
        tools.grid(row=2, column=2, columnspan=3, sticky="w", pady=(3, 0))
        self.review_toggle = ttk.Checkbutton(tools, text="相似鸟复核（实验建议，默认关闭）",
                                             variable=self.review_enabled)
        self.review_toggle.pack(side="left")
        ttk.Style(self.window).configure("Review.TButton", padding=(7, 0), font=("Microsoft YaHei UI", 9))
        self.review_flow_button = ttk.Button(tools, text="复核流程 / 模块状态", style="Review.TButton",
                                             command=self.show_review_flow)
        self.review_flow_button.pack(side="left", padx=(8, 0))
        self.review_label = ttk.Label(self.candidate_label.master, textvariable=self.review_text,
                                      wraplength=350, justify="left", foreground="#805500")
        self.review_label.pack(fill="x", pady=(10, 6))
        self.review_label.bind("<Configure>", lambda e: self.review_label.configure(wraplength=max(120, e.width - 8)))

    def controls(self, busy):
        super().controls(busy)
        self.review_toggle.configure(state="disabled" if busy or self.closing else "normal")

    def start_batch(self):
        if self.busy or self.closing:
            return
        self.batch_review_enabled = bool(self.review_enabled.get())
        if self.backend is not None:
            self.backend.set_enabled(self.batch_review_enabled)
        super().start_batch()

    def clear_birds(self):
        super().clear_birds()
        self.review_text.set("选择一只鸟查看独立复核；原表格和前五候选保持。")

    def show_job(self, key):
        super().show_job(key)
        if self.current_result and not self.current_result.get("boxes"):
            self.review_text.set("没有最终鸟框可供复核；复核不会删除鸟框或把组外等同无鸟。")

    def select_bird(self, event=None):
        super().select_bird(event)
        chosen = self.bird_tree.selection()
        if not chosen or not self.current_result:
            return
        box = self.current_result["boxes"][int(chosen[0])]
        review = self.current_result.get("similar_species_review", {})
        row = matching_review_row(box, review.get("rows", []))
        lines = ["独立复核 · 第 " + str(int(chosen[0]) + 1) + " 只（不改原判）"]
        info = box.get("species_info", {})
        lines.append("原判：" + str(info.get("display_name") or info.get("common_name") or info.get("name") or "待确认"))
        if row is None:
            lines.append(review.get("message") or "未取得可唯一关联到此鸟的复核记录。")
        else:
            names = {"disabled": "本批复核关闭", "not_supported": "N 对照不支持此复核",
                     "no_advice": "该组未启用建议", "group_inactive_no_advice": "该组未启用建议",
                     "not_enabled": "该组未启用建议", "not_triggered": "未触发相似鸟复核",
                     "invalid_input": "输入无法安全关联，保持原判", "routed": "进入独立实验复核",
                     "not_routed": "未进入专项复核", "group_conflict": "多个组冲突，弃权", "conflict": "多个组冲突，弃权",
                     "reviewer_abstained": "专家弃权", "advice": "独立建议", "suggestion": "独立建议",
                     "outgroup": "专项判断为当前组外", "uncertain": "已复核，建议证据不足",
                     "unavailable": "复核不可用，保留原结果", "agrees": "复核同意原判"}
            state = row.get("status", row.get("reason", review.get("status", "")))
            lines.append(names.get(state, str(state)))
            if row.get("group_label") or row.get("group_id"):
                lines.append("相似组：" + str(row.get("group_label") or row["group_id"]))
            if row.get("reason"):
                lines.append(str(row["reason"]))
            if row.get("proposal_common_name") or row.get("proposal_scientific_name"):
                lines.append("建议：" + str(row.get("proposal_common_name") or row["proposal_scientific_name"]))
                if row.get("proposal_scientific_name"):
                    lines.append(str(row["proposal_scientific_name"]))
                if row.get("conditional_confidence") is not None:
                    lines.append(f"组内条件分数 {float(row['conditional_confidence']):.3f}")
                if row.get("group_mass") is not None:
                    lines.append(f"组目标分数质量 {float(row['group_mass']):.3f}；不等于全词表置信度。")
        lines.append("OTHER 仅表示当前相似组外；保留此鸟框、原候选和确认状态。")
        self.review_text.set("\n".join(lines))

    def show_review_flow(self):
        window = tk.Toplevel(self.window)
        window.title("复核流程 / 模块状态")
        window.geometry("730x590")
        text = tk.Text(window, wrap="word", padx=18, pady=16, font=("Microsoft YaHei UI", 11))
        text.pack(fill="both", expand=True)
        text.insert("1.0", "当前实际流程\n\n原图\n  ↓ 第四版 YOLO：普通整图 / 分块补检\n"
                    "每只鸟分别裁切\n  ↓ 原双 BioCLIP 识别鸟种\n"
                    "查看第一候选\n  ├─ 未开启复核 → 保留原判\n"
                    "  ├─ 第一候选不是目标柳莺 → 保留原判\n"
                    "  └─ 已开启，且第一候选是4种目标柳莺之一 → 柳莺实验复核\n\n"
                    "复核后\n• 一致：标记复核同意。\n• 看不清、组外或证据不足：弃权，保留原判。\n"
                    "• 有不同意见且满足原冻结规则：单独显示建议，不自动修改种名。\n"
                    "• 多个复核组冲突：弃权，保留原判。\n\n"
                    "模块状态\n• 柳莺：原冻结实验建议，默认关闭；原模型高置信也可以复核。\n"
                    "• 鸫类 / 麻雀：未启用（no_advice），继续原判。\n"
                    "• 上一版 N：保留原流程，不运行此复核。\n\n"
                    "每批开关固定，处理中不可更改。复核不删鸟框，不覆盖原种名、前五候选或确认状态。"
                    "小鸟和远处的鸟也保留。OTHER 仅指这4种目标柳莺之外，不是没有鸟。")
        text.configure(state="disabled")
        self.modal_windows.append(window)
