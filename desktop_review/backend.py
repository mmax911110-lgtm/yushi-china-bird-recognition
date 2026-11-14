"""Per-crop advisory adapter around the existing, unmodified desktop engine.

The only enabled specialist is the previously frozen willow R2 experiment.
Its original suggestion gates remain unchanged. The new router is an entry
gate, not permission to bypass the specialist's frozen evidence thresholds.
"""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import hashlib
import html
import json
import time

import numpy as np

from desktop_review.routing import GROUPS, route_candidate


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def image_sha(image):
    stream = BytesIO()
    image.save(stream, format="PNG")
    return hashlib.sha256(stream.getvalue()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


class CaptureMixin:
    """Capture the exact original H features, including each tile's offsets.

    A method wrapper exists only on this encoder instance during the original
    forward; it is always restored. No additional visual inference is run.
    """
    def _apply_selected(self, original, result):
        infos = [b["species_info"] for b in original["boxes"]]
        indices = [int(i["feature_index"]) for i in infos]
        block = {"crop_count": len(original["_crops"]), "indices": indices,
                 "values": None, "hashes": [image_sha(original["_crops"][i]) for i in indices]}
        if self.mode != "selected" or not infos:
            answer = super()._apply_selected(original, result)
            self._review_blocks.append(block)
            return answer
        if self.p_encoder is None:
            self.base.encoder.sync()
            self.base.encoder.model.to("cpu")
            self._empty_cache()
            from inference.encoders import LocalEncoder
            self.p_encoder = LocalEncoder(self.root, "bioclip25", self.device, 2, self.numeric_mode)
        encoder = self.p_encoder
        function = encoder.encode_images
        sentinel = object()
        previous = encoder.__dict__.get("encode_images", sentinel)

        def capture(crops):
            answer = function(crops)
            try:
                require(block["values"] is None, "Original classifier encoded a batch twice")
                require([image_sha(c) for c in crops] == block["hashes"], "Original crop order changed")
                values = np.asarray(answer[0], np.float32)
                require(values.shape == (len(indices), 1024) and np.isfinite(values).all(),
                        "Invalid captured BioCLIP2.5 features")
                block["values"] = values.copy()
            except Exception as exc:
                block["capture_error"] = str(exc)
            return answer

        encoder.encode_images = capture
        try:
            answer = super()._apply_selected(original, result)
        finally:
            if previous is sentinel:
                del encoder.encode_images
            else:
                encoder.encode_images = previous
        if block["values"] is None:
            block.setdefault("capture_error", "No original H forward captured")
        self._review_blocks.append(block)
        return answer

    def predict(self, image, detection_mode="normal", **kwargs):
        self._review_blocks = []
        answer = super().predict(image, detection_mode=detection_mode, **kwargs)
        if self.mode != "selected":
            return answer
        try:
            offset, mapping, order = 0, {}, []
            for block in self._review_blocks:
                require(not block.get("capture_error"), block.get("capture_error"))
                indices = [offset + i for i in block["indices"]]
                require(not indices or block["values"] is not None, "Missing original final-crop features")
                for i, fi in enumerate(indices):
                    require(image_sha(answer["_crops"][fi]) == block["hashes"][i], "Tile feature/crop mismatch")
                    require(fi not in mapping, "Repeated final feature index")
                    mapping[fi] = block["values"][i].copy()
                order.extend(indices)
                offset += block["crop_count"]
            require(offset == len(answer["_crops"]), "Tile crop offsets changed")
            require(order == [int(b["species_info"]["feature_index"]) for b in answer["boxes"]],
                    "Final bird order changed")
            answer["_review_bio25"] = mapping
        except Exception as exc:
            answer["_review_capture_error"] = str(exc)
            answer["_review_bio25"] = {}
        return answer


class WillowHead:
    """Inference-only NumPy reproduction of the frozen species-aware R2 head."""
    def __init__(self, package_root, research_root):
        folder = Path(research_root) / "species_rejection_20261114"
        self.frozen_path = folder / "frozen_expert.json"
        frozen = read(self.frozen_path)
        require(frozen.get("ready") and frozen.get("selection_frozen_before_test_scoring")
                and frozen.get("deployment_mode") == "advisory_only", "柳莺冻结配置不可用")
        # Resolve by known role, not historical machine-specific absolute paths.
        head_path = folder / "heads/species_aware_OTHER_bio25_C0p1/head.npz"
        decision_path, policy_path = folder / "frozen_decision.json", folder / "frozen_review_policy.json"
        require(sha(head_path) == "0745aa534bd75c331b3977f099bcd60c9637f7d9b4ae35ac2b47608a487fc6c6",
                "柳莺权重与冻结版本不一致")
        require(sha(decision_path) == frozen["decision_spec_sha256"] and
                sha(policy_path) == frozen["review_policy_sha256"], "柳莺冻结规则发生变化")
        require(sha(Path(package_root) / "configs/china_labels.json") == frozen["labels_sha256"],
                "鸟种索引与复核权重不一致")
        main_binding = frozen["original_main_assets"]
        config_path = Path(package_root) / "configs/recognizer.json"
        require(sha(config_path) == main_binding["configuration_sha256"] and
                read(config_path)["asset_hashes"] == main_binding["asset_hashes"],
                "当前主网络与柳莺冻结特征空间不一致")
        # The original engine verifies these actual model files before any
        # visual forward. The specialist additionally checks the pinned map,
        # preventing a different valid package from silently changing space.
        decision, policy = read(decision_path), read(policy_path)
        require(policy["apply_overrides"] is False and decision["threshold"] == frozen["threshold"] == 0.,
                "只支持原冻结的建议模式与OTHER规则")
        self.rule = policy["groups"][0]
        self.targets = [70, 126, 198, 473]
        require(self.rule["global_ids"] == self.targets and self.rule["expert_model_sha256"] == sha(head_path),
                "柳莺组与冻结权重绑定不一致")
        with np.load(head_path, allow_pickle=False) as values:
            self.meta = json.loads(str(values["metadata_json"].item()))
            require(self.meta["parts"] == ["bio25"] and self.meta["structure"] == "species_aware_OTHER",
                    "不支持的柳莺特征分支")
            require(self.meta["labels_sha256"] == frozen["labels_sha256"] and
                    self.meta["binding"]["manifest_sha256"] == frozen["data_manifest_sha256"],
                    "柳莺训练/类别绑定不一致")
            self.coef, self.intercept = values["coef"].copy(), values["intercept"].copy()
            self.mean, self.scale = values["scaler_mean"].copy(), values["scaler_scale"].copy()
            self.classes = values["raw_class_order"].tolist()
        require(self.coef.shape == (len(self.classes), 1024) and self.mean.shape == self.scale.shape == (1024,)
                and (self.scale > 0).all(), "柳莺分类头维度不匹配")
        require(all(np.isfinite(v).all() for v in (self.coef, self.intercept, self.mean, self.scale)),
                "柳莺分类头含非有限数")
        require(self.classes == decision["raw_class_order"], "柳莺类别顺序不匹配")
        self.binding = {"head_sha256": sha(head_path), "frozen_sha256": sha(self.frozen_path),
                        "policy_sha256": sha(policy_path), "decision_sha256": sha(decision_path)}

    def predict(self, feature):
        x = np.asarray(feature, np.float64).reshape(1, -1)
        require(x.shape == (1, 1024) and np.isfinite(x).all(), "柳莺输入特征无效")
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        require((norm > 1e-12).all(), "柳莺输入为零向量")
        logits = ((x / norm - self.mean) / self.scale) @ self.coef.T + self.intercept
        raw = softmax(logits)
        columns = [self.classes.index(i) for i in self.targets]
        target = raw[:, columns]
        other = np.maximum(0., 1 - target.sum(1))
        full = np.column_stack((other, target))
        full /= full.sum(1, keepdims=True)
        q = softmax(logits[:, columns])[0]
        order = np.argsort(-q, kind="stable")
        return {"rejected_as_OTHER": bool(full[0].argmax() == 0),
                "candidate_global_id": self.targets[int(order[0])],
                "conditional_confidence": float(q[order[0]]),
                "conditional_margin": float(q[order[0]] - q[order[1]]),
                "group_mass": float(full[0, 1:].sum()), "conditional_probabilities": q.tolist()}


def softmax(logits):
    values = np.asarray(logits, np.float64)
    exp = np.exp(values - values.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


class ReviewBackend:
    def __init__(self, package_root, research_root, output_root):
        self.package_root, self.research_root = Path(package_root), Path(research_root)
        self.output_root = Path(output_root)
        self.enabled = False
        self.engine = self.head = None
        self.labels = read(self.package_root / "configs/china_labels.json")
        self.names = {r["scientific_name"]: i for i, r in enumerate(self.labels)}
        require(len(self.labels) == len(self.names) == 1343, "原1343鸟种目录不完整")

    def set_enabled(self, enabled):
        # Set only on the GUI thread before a whole batch starts, not mid-batch.
        self.enabled = bool(enabled)

    def factory(self, root, device="auto", mode="selected", progress=None):
        from engine import BirdRecognizer
        class ReviewedRecognizer(CaptureMixin, BirdRecognizer):
            pass
        self.engine = ReviewedRecognizer(root, device=device, mode=mode, progress=progress)
        return self.engine

    def _review_row(self, image, result, public_box, index):
        box = result["boxes"][index]
        info = box["species_info"]
        top = info.get("top5", [])[:5]
        ids = [self.names.get(p.get("name"), -1) for p in top]
        route = route_candidate(ids, enabled=self.enabled)
        row = {**route, "reason_code": route["reason"], "reason": route.get("message", route["reason"]),
               "bird_id": public_box["bird_id"], "feature_index": info["feature_index"],
               "xyxy": deepcopy(box["xyxy"]), "detection_confidence": box.get("confidence"),
               "original_display_name": info["display_name"], "original_accepted": bool(info["accepted"]),
               "original_confidence": float(info["confidence"]), "crop_sha256": public_box["crop_sha256"],
               "proposal_global_id": -1, "proposal_common_name": None, "applied_to_original": False}
        if route["status"] != "routed":
            return row
        if getattr(self.engine, "mode", "selected") != "selected":
            return {**row, "status": "not_supported", "reason": "上一版N保留原流程；复核需要双BioCLIP特征"}
        try:
            from inference.engine_n import square_context
            require(not result.get("_review_capture_error"), result.get("_review_capture_error"))
            fi = int(info["feature_index"])
            crop = result["_crops"][fi]
            require(image_sha(crop) == row["crop_sha256"] == image_sha(square_context(image, box["xyxy"])),
                    "鸟框、原图与实际裁图绑定不一致")
            feature = result["_review_bio25"][fi]
            if self.head is None:
                self.head = WillowHead(self.package_root, self.research_root)
            pred = self.head.predict(feature)
            row.update(pred, binding=self.head.binding)
            if pred["rejected_as_OTHER"]:
                return {**row, "status": "outgroup", "reason": "不属于这4种目标柳莺；保留鸟框和原结果"}
            # Preserve all historical species-advice gates, even though the
            # new entry router also permits confident group members to review.
            rule = self.head.rule
            evidence = (len(ids) >= 2 and ids[0] in self.head.targets and ids[1] in self.head.targets
                        and float(top[0]["confidence"]) - float(top[1]["confidence"]) <= rule["trigger_max_margin"]
                        and float(top[0]["confidence"]) + float(top[1]["confidence"]) >= rule["trigger_min_pair_mass"]
                        and pred["conditional_confidence"] >= rule["expert_min_confidence"]
                        and pred["conditional_margin"] >= rule["expert_min_margin"]
                        and pred["group_mass"] >= rule["expert_min_domain_score"]
                        and pred["candidate_global_id"] in ids[:2])
            if not evidence:
                return {**row, "status": "uncertain", "reason": "已复核，未满足原冻结的种名建议条件；保持原判"}
            label = self.labels[pred["candidate_global_id"]]
            return {**row, "status": "agrees" if pred["candidate_global_id"] == ids[0] else "suggestion",
                    "reason": "复核一致" if pred["candidate_global_id"] == ids[0] else "复核有不同建议，原结果保留",
                    "proposal_global_id": pred["candidate_global_id"],
                    "proposal_common_name": label.get("common_name") or label["scientific_name"],
                    "proposal_scientific_name": label["scientific_name"]}
        except Exception as exc:
            return {**row, "status": "unavailable", "reason": "复核不可用，原结果保留：" + str(exc)}

    def save(self, image, result, source, output_root):
        from engine import save_result
        protected = digest(result["boxes"])
        public = save_result(image, result, source, output_root)
        original_sha = sha(public["result_json"])
        start = time.perf_counter()
        rows = []
        for i, box in enumerate(public["boxes"]):
            try:
                rows.append(self._review_row(image, result, box, i))
            except Exception as exc:
                rows.append({"bird_id": box.get("bird_id"), "feature_index": box.get("species_info", {}).get("feature_index"),
                             "xyxy": box.get("xyxy"), "original_display_name": box.get("species_info", {}).get("display_name", "鸟"),
                             "status": "unavailable", "reason": "复核异常，保留已完成的原识别：" + str(exc),
                             "proposal_global_id": -1, "proposal_common_name": None, "applied_to_original": False})
        require(digest(result["boxes"]) == protected and sha(public["result_json"]) == original_sha,
                "Independent review unexpectedly changed the original recognition")
        review = {"status": "reviewed" if self.enabled else "disabled", "enabled": self.enabled,
                  "message": "逐鸟复核仅给独立建议；鸫类、麻雀未启用。" if self.enabled else "相似种复核已关闭。",
                  "rows": rows, "groups": deepcopy(GROUPS), "original_result_sha256": original_sha,
                  "original_boxes_sha256": protected, "applied_to_original": False,
                  "additional_visual_forwards": 0, "extra_review_ms": (time.perf_counter() - start) * 1000,
                  "routing_version": "top1_or_registered_confusion_v1", "confusion_neighbors": {},
                  "confidence_scope": "specialist conditional within group, not full1343 probability"}
        folder = Path(public["event_dir"])
        review_path = folder / "similar_species_review.json"
        try:
            review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            cards = []
            for box, row in zip(public["boxes"], rows):
                cells = (row["bird_id"], row["original_display_name"], row["reason"], row.get("proposal_common_name") or "—")
                cards.append("<tr>" + "".join("<td>" + html.escape(str(c)) + "</td>" for c in cells) + "</tr>")
            (folder / "similar_species_review.html").write_text(
                '<!doctype html><meta charset="utf-8"><title>逐鸟相似种复核</title><h1>原结果与独立建议</h1>'
                '<p>复核不修改原鸟框、Top5与确认状态。OTHER只指当前组外。</p>'
                '<table border="1"><tr><th>鸟框</th><th>原识别</th><th>复核状态</th><th>独立建议</th></tr>'
                + "".join(cards) + "</table>", encoding="utf-8")
        except Exception as exc:
            review["export_error"] = str(exc)
            review["message"] += " 独立复核报告保存失败；原识别与裁图已保存。"
        # The original result.json is immutable. Only the worker's in-memory
        # public copy gains a separate UI field; diagnostics live in a sidecar.
        public["similar_species_review"] = review
        public["review_json"] = str(review_path) if review_path.is_file() else None
        return public
