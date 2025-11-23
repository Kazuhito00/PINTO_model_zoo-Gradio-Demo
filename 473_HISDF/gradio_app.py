#!/usr/bin/env python

from __future__ import annotations
import warnings

warnings.filterwarnings("ignore")
import os
import copy
import cv2
import math
import time
from pprint import pprint
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
from collections import Counter
from abc import ABC, abstractmethod
import threading
import tempfile
import gradio as gr

# Set Gradio language to English
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_DEFAULT_DIR"] = "."

# ============================================================================
# Constants and Classes from demo_deimv2_onnx_wholebody34_with_edges.py
# ============================================================================

AVERAGE_HEAD_WIDTH: float = 0.16 + 0.10  # 16cm + Margin Compensation

BOX_COLORS = [
    [(216, 67, 21), "Front"],
    [(255, 87, 34), "Right-Front"],
    [(123, 31, 162), "Right-Side"],
    [(255, 193, 7), "Right-Back"],
    [(76, 175, 80), "Back"],
    [(33, 150, 243), "Left-Back"],
    [(156, 39, 176), "Left-Side"],
    [(0, 188, 212), "Left-Front"],
]

# The pairs of classes you want to join
EDGES = [
    (21, 22),
    (21, 22),  # collarbone -> shoulder (left and right)
    (21, 23),  # collarbone -> solar_plexus
    (22, 24),
    (22, 24),  # shoulder -> elbow (left and right)
    (22, 30),
    (22, 30),  # shoulder -> hip_joint (left and right)
    (24, 25),
    (24, 25),  # elbow -> wrist (left and right)
    (23, 29),  # solar_plexus -> abdomen
    (29, 30),
    (29, 30),  # abdomen -> hip_joint (left and right)
    (30, 31),
    (30, 31),  # hip_joint -> knee (left and right)
    (31, 32),
    (31, 32),  # knee -> ankle (left and right)
]


@dataclass(frozen=False)
class Box:
    classid: int
    score: float
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int
    cz: int
    mask: np.ndarray
    generation: int = -1  # -1: Unknown, 0: Adult, 1: Child
    gender: int = -1  # -1: Unknown, 0: Male, 1: Female
    handedness: int = -1  # -1: Unknown, 0: Left, 1: Right
    head_pose: int = -1  # -1: Unknown, 0-7: Various poses
    is_used: bool = False
    person_id: int = -1
    track_id: int = -1


class SimpleSortTracker:
    """Minimal SORT-style tracker based on IoU matching."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_track_id = 1
        self.tracks: List[Dict[str, Any]] = []
        self.frame_index = 0

    @staticmethod
    def _iou(
        bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]
    ) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        if inter_w == 0 or inter_h == 0:
            return 0.0

        inter_area = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return float(inter_area / union)

    def update(self, boxes: List[Box]) -> None:
        self.frame_index += 1

        for box in boxes:
            box.track_id = -1

        if not boxes and not self.tracks:
            return

        iou_matrix = None
        if self.tracks and boxes:
            iou_matrix = np.zeros((len(self.tracks), len(boxes)), dtype=np.float32)
            for t_idx, track in enumerate(self.tracks):
                track_bbox = track["bbox"]
                for d_idx, box in enumerate(boxes):
                    det_bbox = (box.x1, box.y1, box.x2, box.y2)
                    iou_matrix[t_idx, d_idx] = self._iou(track_bbox, det_bbox)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        matches: List[Tuple[int, int]] = []

        if iou_matrix is not None and iou_matrix.size > 0:
            while True:
                best_track = -1
                best_det = -1
                best_iou = self.iou_threshold
                for t_idx in range(len(self.tracks)):
                    if t_idx in matched_tracks:
                        continue
                    for d_idx in range(len(boxes)):
                        if d_idx in matched_detections:
                            continue
                        iou = float(iou_matrix[t_idx, d_idx])
                        if iou > best_iou:
                            best_iou = iou
                            best_track = t_idx
                            best_det = d_idx
                if best_track == -1:
                    break
                matched_tracks.add(best_track)
                matched_detections.add(best_det)
                matches.append((best_track, best_det))

        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            det_box = boxes[d_idx]
            track["bbox"] = (det_box.x1, det_box.y1, det_box.x2, det_box.y2)
            track["missed"] = 0
            track["last_seen"] = self.frame_index
            det_box.track_id = track["id"]

        surviving_tracks: List[Dict[str, Any]] = []
        for idx, track in enumerate(self.tracks):
            if idx in matched_tracks:
                surviving_tracks.append(track)
                continue
            track["missed"] += 1
            if track["missed"] <= self.max_age:
                surviving_tracks.append(track)
        self.tracks = surviving_tracks

        for d_idx, det_box in enumerate(boxes):
            if d_idx in matched_detections:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            det_box.track_id = track_id
            self.tracks.append(
                {
                    "id": track_id,
                    "bbox": (det_box.x1, det_box.y1, det_box.x2, det_box.y2),
                    "missed": 0,
                    "last_seen": self.frame_index,
                }
            )


class AbstractModel(ABC):
    """Base class of the model."""

    _runtime: str = "onnx"
    _model_path: str = ""
    _obj_class_score_th: float = 0.35
    _attr_class_score_th: float = 0.70
    _input_names: List[str] = []
    _output_names: List[str] = []
    _interpreter = None
    _providers = None
    _swap = (2, 0, 1)
    _onnx_dtypes_to_np_dtypes = {
        "tensor(float)": np.float32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
    }

    @abstractmethod
    def __init__(
        self,
        *,
        runtime: Optional[str] = "onnx",
        model_path: Optional[str] = "",
        obj_class_score_th: Optional[float] = 0.35,
        attr_class_score_th: Optional[float] = 0.70,
        keypoint_th: Optional[float] = 0.25,
        providers: Optional[List] = None,
    ):
        self._runtime = runtime
        self._model_path = model_path
        self._obj_class_score_th = obj_class_score_th
        self._attr_class_score_th = attr_class_score_th
        self._keypoint_th = keypoint_th
        self._providers = providers

        if self._runtime == "onnx":
            import onnxruntime

            onnxruntime.set_default_logger_severity(3)
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3
            self._interpreter = onnxruntime.InferenceSession(
                model_path,
                sess_options=session_option,
                providers=providers,
            )
            self._providers = self._interpreter.get_providers()
            print("Enabled ONNX ExecutionProviders:")
            pprint(self._providers)

            self._input_names = [input.name for input in self._interpreter.get_inputs()]
            self._input_dtypes = [
                self._onnx_dtypes_to_np_dtypes[input.type]
                for input in self._interpreter.get_inputs()
            ]
            self._output_names = [
                output.name for output in self._interpreter.get_outputs()
            ]
            self._model = self._interpreter.run
            self._swap = (2, 0, 1)

    @abstractmethod
    def __call__(
        self,
        *,
        input_datas: List[np.ndarray],
    ) -> List[np.ndarray]:
        datas = {
            f"{input_name}": input_data
            for input_name, input_data in zip(self._input_names, input_datas)
        }
        if self._runtime == "onnx":
            outputs = [
                output
                for output in self._model(
                    output_names=self._output_names,
                    input_feed=datas,
                )
            ]
            return outputs

    @abstractmethod
    def _preprocess(
        self,
        *,
        image: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _postprocess(
        self,
        *,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        raise NotImplementedError()


class HISDF(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = "onnx",
        model_path: Optional[str] = "",
        obj_class_score_th: Optional[float] = 0.35,
        attr_class_score_th: Optional[float] = 0.70,
        keypoint_th: Optional[float] = 0.35,
        providers: Optional[List] = None,
    ):
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            obj_class_score_th=obj_class_score_th,
            attr_class_score_th=attr_class_score_th,
            keypoint_th=keypoint_th,
            providers=providers,
        )
        self.mean: np.ndarray = np.asarray(
            [0.485, 0.456, 0.406], dtype=np.float32
        ).reshape([3, 1, 1])
        self.std: np.ndarray = np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        ).reshape([3, 1, 1])

    def __call__(
        self,
        image: np.ndarray,
        disable_generation_identification_mode: bool,
        disable_gender_identification_mode: bool,
        disable_left_and_right_hand_identification_mode: bool,
        disable_headpose_identification_mode: bool,
    ) -> Tuple[List[Box], np.ndarray, np.ndarray]:
        temp_image = copy.deepcopy(image)
        resized_image = self._preprocess(temp_image)
        inferece_image = np.asarray([resized_image], dtype=self._input_dtypes[0])
        outputs = super().__call__(input_datas=[inferece_image])
        bbox_classid_xyxy_score, depth, binary_masks, instance_masks = outputs[0], outputs[1], outputs[2], outputs[3]
        result_boxes, result_depth, result_seg = self._postprocess(
            image=temp_image,
            boxes=bbox_classid_xyxy_score,
            depth=depth,
            segment=binary_masks,
            instance_segment=instance_masks,
            disable_generation_identification_mode=disable_generation_identification_mode,
            disable_gender_identification_mode=disable_gender_identification_mode,
            disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
            disable_headpose_identification_mode=disable_headpose_identification_mode,
        )
        return result_boxes, result_depth, result_seg

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        image = image.transpose(self._swap)
        image = np.ascontiguousarray(image, dtype=np.float32)
        return image

    def _postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        depth: np.ndarray,
        segment: np.ndarray,
        instance_segment: np.ndarray,
        disable_generation_identification_mode: bool,
        disable_gender_identification_mode: bool,
        disable_left_and_right_hand_identification_mode: bool,
        disable_headpose_identification_mode: bool,
    ) -> Tuple[List[Box], np.ndarray, np.ndarray]:
        image_height = image.shape[0]
        image_width = image.shape[1]
        result_boxes: List[Box] = []
        box_score_threshold: float = min(
            [self._obj_class_score_th, self._attr_class_score_th, self._keypoint_th]
        )

        # 0.0-1.0 -> 0 - 255
        min_val: np.ndarray = depth.min()
        max_val: np.ndarray = depth.max()
        depth = depth.squeeze()
        result_depth = ((depth - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        result_seg = segment.squeeze()

        if len(boxes) > 0:
            scores = boxes[:, 5:6]
            keep_idxs = scores[:, 0] > box_score_threshold
            keep_indices = np.where(keep_idxs)[0]
            scores_keep = scores[keep_idxs, :]
            boxes_keep = boxes[keep_idxs, :]

            body_indices = np.where(boxes[:, 0].astype(int) == 0)[0]
            instance_segment_map: Dict[int, np.ndarray] = {}
            if (
                instance_segment is not None and
                isinstance(instance_segment, np.ndarray) and
                instance_segment.ndim >= 3 and
                instance_segment.size > 0
            ):
                available_masks = min(len(body_indices), instance_segment.shape[0])
                if available_masks > 0:
                    body_masks = instance_segment[:available_masks, 0, ...]
                    for idx, body_idx in enumerate(body_indices[:available_masks]):
                        instance_segment_map[int(body_idx)] = body_masks[idx]

            if len(boxes_keep) > 0:
                for keep_idx, box, score in zip(keep_indices, boxes_keep, scores_keep):
                    classid = int(box[0])
                    x_min = int(max(0, box[1]) * image_width)
                    y_min = int(max(0, box[2]) * image_height)
                    x_max = int(min(box[3], 1.0) * image_width)
                    y_max = int(min(box[4], 1.0) * image_height)
                    cx = (x_min + x_max) // 2
                    cy = (y_min + y_max) // 2
                    cz = 0.0
                    crx1, crx2 = np.clip([cx - 3, cx + 3], 0, image_width)
                    cry1, cry2 = np.clip([cy - 3, cy + 3], 0, image_height)
                    result_boxes.append(
                        Box(
                            classid=classid,
                            score=float(score),
                            x1=x_min,
                            y1=y_min,
                            x2=x_max,
                            y2=y_max,
                            cx=cx,
                            cy=cy,
                            cz=np.median(result_depth[cry1:cry2, crx1:crx2]),
                            mask=None if classid != 0 else instance_segment_map.get(int(keep_idx)),
                        )
                    )

                result_boxes = [
                    box
                    for box in result_boxes
                    if (
                        box.classid in [0, 5, 6, 7, 16, 17, 18, 19, 20, 26, 27, 28, 33]
                        and box.score >= self._obj_class_score_th
                    )
                    or box.classid
                    not in [0, 5, 6, 7, 16, 17, 18, 19, 20, 26, 27, 28, 33]
                ]
                result_boxes = [
                    box
                    for box in result_boxes
                    if (
                        box.classid in [1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15]
                        and box.score >= self._attr_class_score_th
                    )
                    or box.classid not in [1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15]
                ]
                result_boxes = [
                    box
                    for box in result_boxes
                    if (
                        box.classid in [21, 22, 23, 24, 25, 29, 30, 31, 32]
                        and box.score >= self._keypoint_th
                    )
                    or box.classid not in [21, 22, 23, 24, 25, 29, 30, 31, 32]
                ]

                if not disable_generation_identification_mode:
                    body_boxes = [box for box in result_boxes if box.classid == 0]
                    generation_boxes = [
                        box for box in result_boxes if box.classid in [1, 2]
                    ]
                    self._find_most_relevant_obj(
                        base_objs=body_boxes, target_objs=generation_boxes
                    )
                result_boxes = [
                    box for box in result_boxes if box.classid not in [1, 2]
                ]

                if not disable_gender_identification_mode:
                    body_boxes = [box for box in result_boxes if box.classid == 0]
                    gender_boxes = [
                        box for box in result_boxes if box.classid in [3, 4]
                    ]
                    self._find_most_relevant_obj(
                        base_objs=body_boxes, target_objs=gender_boxes
                    )
                result_boxes = [
                    box for box in result_boxes if box.classid not in [3, 4]
                ]

                if not disable_headpose_identification_mode:
                    head_boxes = [box for box in result_boxes if box.classid == 7]
                    headpose_boxes = [
                        box
                        for box in result_boxes
                        if box.classid in [8, 9, 10, 11, 12, 13, 14, 15]
                    ]
                    self._find_most_relevant_obj(
                        base_objs=head_boxes, target_objs=headpose_boxes
                    )
                result_boxes = [
                    box
                    for box in result_boxes
                    if box.classid not in [8, 9, 10, 11, 12, 13, 14, 15]
                ]

                if not disable_left_and_right_hand_identification_mode:
                    hand_boxes = [box for box in result_boxes if box.classid == 26]
                    left_right_hand_boxes = [
                        box for box in result_boxes if box.classid in [27, 28]
                    ]
                    self._find_most_relevant_obj(
                        base_objs=hand_boxes, target_objs=left_right_hand_boxes
                    )
                result_boxes = [
                    box for box in result_boxes if box.classid not in [27, 28]
                ]

                for target_classid in [21, 22, 23, 24, 25, 29, 30, 31, 32]:
                    keypoints_boxes = [
                        box for box in result_boxes if box.classid == target_classid
                    ]
                    filtered_keypoints_boxes = self._nms(
                        target_objs=keypoints_boxes, iou_threshold=0.20
                    )
                    result_boxes = [
                        box for box in result_boxes if box.classid != target_classid
                    ]
                    result_boxes = result_boxes + filtered_keypoints_boxes

        return result_boxes, result_depth, result_seg

    def _find_most_relevant_obj(
        self,
        *,
        base_objs: List[Box],
        target_objs: List[Box],
    ):
        for base_obj in base_objs:
            most_relevant_obj: Box = None
            best_score = 0.0
            best_iou = 0.0
            best_distance = float("inf")

            for target_obj in target_objs:
                distance = (
                    (base_obj.cx - target_obj.cx) ** 2
                    + (base_obj.cy - target_obj.cy) ** 2
                ) ** 0.5
                if not target_obj.is_used and distance <= 10.0:
                    if target_obj.score >= best_score:
                        iou: float = self._calculate_iou(
                            base_obj=base_obj, target_obj=target_obj
                        )
                        if iou > best_iou:
                            most_relevant_obj = target_obj
                            best_iou = iou
                            best_distance = distance
                            best_score = target_obj.score
                        elif iou > 0.0 and iou == best_iou:
                            if distance < best_distance:
                                most_relevant_obj = target_obj
                                best_distance = distance
                                best_score = target_obj.score
            if most_relevant_obj:
                if most_relevant_obj.classid == 1:
                    base_obj.generation = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 2:
                    base_obj.generation = 1
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 3:
                    base_obj.gender = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 4:
                    base_obj.gender = 1
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 8:
                    base_obj.head_pose = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 9:
                    base_obj.head_pose = 1
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 10:
                    base_obj.head_pose = 2
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 11:
                    base_obj.head_pose = 3
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 12:
                    base_obj.head_pose = 4
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 13:
                    base_obj.head_pose = 5
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 14:
                    base_obj.head_pose = 6
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 15:
                    base_obj.head_pose = 7
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 27:
                    base_obj.handedness = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 28:
                    base_obj.handedness = 1
                    most_relevant_obj.is_used = True

    def _nms(
        self,
        *,
        target_objs: List[Box],
        iou_threshold: float,
    ):
        filtered_objs: List[Box] = []
        sorted_objs = sorted(target_objs, key=lambda box: box.score, reverse=True)

        while sorted_objs:
            current_box = sorted_objs.pop(0)
            if current_box.is_used:
                continue
            filtered_objs.append(current_box)
            current_box.is_used = True

            remaining_boxes = []
            for box in sorted_objs:
                if not box.is_used:
                    iou_value = self._calculate_iou(
                        base_obj=current_box, target_obj=box
                    )
                    if iou_value >= iou_threshold:
                        box.is_used = True
                    else:
                        remaining_boxes.append(box)
            sorted_objs = remaining_boxes
        return filtered_objs

    def _calculate_iou(
        self,
        *,
        base_obj: Box,
        target_obj: Box,
    ) -> float:
        inter_xmin = max(base_obj.x1, target_obj.x1)
        inter_ymin = max(base_obj.y1, target_obj.y1)
        inter_xmax = min(base_obj.x2, target_obj.x2)
        inter_ymax = min(base_obj.y2, target_obj.y2)
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        area1 = (base_obj.x2 - base_obj.x1) * (base_obj.y2 - base_obj.y1)
        area2 = (target_obj.x2 - target_obj.x1) * (target_obj.y2 - target_obj.y1)
        iou = inter_area / float(area1 + area2 - inter_area)
        return iou


def get_id_color(index) -> List[int]:
    temp_index = abs(int(index + 5)) * 3
    color = [(37 * temp_index) % 255, (17 * temp_index) % 255, (29 * temp_index) % 255]
    return color


def draw_skeleton(
    image: np.ndarray,
    boxes: List[Box],
    color=(0, 255, 255),
):
    """Draw skeleton connections between keypoints using 3D distance."""
    # Assign person IDs
    person_boxes = [b for b in boxes if b.classid == 0]
    for i, pbox in enumerate(person_boxes):
        pbox.person_id = i

    # Assign person IDs to keypoints
    keypoint_ids = {21, 22, 23, 24, 25, 29, 30, 31, 32}
    for box in boxes:
        if box.classid in keypoint_ids:
            box.person_id = -1
            for pbox in person_boxes:
                if (pbox.x1 <= box.cx <= pbox.x2) and (pbox.y1 <= box.cy <= pbox.y2):
                    box.person_id = pbox.person_id
                    break

    # Group boxes by class ID
    classid_to_boxes: Dict[int, List[Box]] = {}
    for b in boxes:
        classid_to_boxes.setdefault(b.classid, []).append(b)

    edge_counts = Counter(EDGES)
    lines_to_draw = []

    for (pid, cid), repeat_count in edge_counts.items():
        parent_list = classid_to_boxes.get(pid, [])
        child_list = classid_to_boxes.get(cid, [])

        if not parent_list or not child_list:
            continue

        for_parent = repeat_count if (pid in [21, 29]) else 1
        parent_capacity = [for_parent] * len(parent_list)
        child_used = [False] * len(child_list)

        pair_candidates = []
        for i, pbox in enumerate(parent_list):
            for j, cbox in enumerate(child_list):
                if (
                    (pbox.person_id is not None)
                    and (cbox.person_id is not None)
                    and (pbox.person_id == cbox.person_id)
                ):
                    # Use 3D distance
                    dist_3d = np.sqrt(
                        (pbox.cx - cbox.cx) ** 2
                        + (pbox.cy - cbox.cy) ** 2
                        + (pbox.cz - cbox.cz) ** 2
                    )
                    pair_candidates.append((dist_3d, i, j))

        # Sort by 3D distance
        pair_candidates.sort(key=lambda x: x[0])

        for _, i, j in pair_candidates:
            if parent_capacity[i] > 0 and (not child_used[j]):
                pbox = parent_list[i]
                cbox = child_list[j]
                lines_to_draw.append(((pbox.cx, pbox.cy), (cbox.cx, cbox.cy)))
                parent_capacity[i] -= 1
                child_used[j] = True

    for pt1, pt2 in lines_to_draw:
        cv2.line(image, pt1, pt2, color, thickness=2)


# ============================================================================
# Gradio Application Code
# ============================================================================

# Global model variables
hisdf_model = None
tracker = SimpleSortTracker()
stream_tracker = SimpleSortTracker()  # Separate tracker for streaming
track_color_cache: Dict[int, np.ndarray] = {}  # Track ID to color mapping

# Streaming frame management
stream_lock = threading.Lock()
stream_latest_frame = None
stream_processing = False
stream_latest_output = None


def scan_onnx_models(directory: str = ".") -> List[str]:
    """Scan directory for HISDF ONNX models."""
    hisdf_models = []

    for file in os.listdir(directory):
        if file.endswith(".onnx"):
            # Match pattern: deimv2*_depthanythingv2_instanceseg_*.onnx
            if file.startswith("deimv2") and "_depthanythingv2_instanceseg_" in file:
                hisdf_models.append(file)

    return sorted(hisdf_models)


def initialize_models(
    hisdf_model_path: str = "deimv2x_depthanythingv2_instanceseg_1x3xHxW.onnx",
    execution_provider: str = "cuda",
    force_reload: bool = False,
):
    """Initialize HISDF model."""
    global hisdf_model

    providers = []
    if execution_provider == "cpu":
        providers = ["CPUExecutionProvider"]
    elif execution_provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif execution_provider == "tensorrt":
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": ".",
                    "trt_fp16_enable": True,
                    "trt_op_types_to_exclude": "NonMaxSuppression,NonZero,RoiAlign",
                },
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    status_messages = []

    # Initialize HISDF model
    if (hisdf_model is None or force_reload) and os.path.exists(hisdf_model_path):
        try:
            hisdf_model = HISDF(
                runtime="onnx",
                model_path=hisdf_model_path,
                obj_class_score_th=0.35,
                attr_class_score_th=0.70,
                keypoint_th=0.30,
                providers=providers,
            )
            status_messages.append(f"✓ HISDF model loaded: {hisdf_model_path}")
        except Exception as e:
            status_messages.append(f"✗ Failed to load HISDF model: {str(e)}")
    elif not os.path.exists(hisdf_model_path):
        status_messages.append(f"✗ HISDF model not found: {hisdf_model_path}")

    return "\n".join(status_messages) if status_messages else "Model already loaded"


def process_image(
    image: np.ndarray,
    obj_threshold: float = 0.35,
    attr_threshold: float = 0.70,
    keypoint_threshold: float = 0.30,
    enable_skeleton: bool = True,
    enable_depth_map_overlay: bool = False,
    enable_instance_segmentation_overlay: bool = False,
    enable_gender: bool = True,
    enable_generation: bool = True,
    enable_headpose: bool = True,
    enable_hand_lr: bool = True,
    enable_tracking: bool = False,
    enable_face_mosaic: bool = False,
    enable_distance: bool = True,
    keypoint_mode: str = "dot",
    camera_fov: int = 90,
    use_stream_tracker: bool = False,
) -> Tuple[np.ndarray, str, float]:
    """Process image with HISDF model."""
    global hisdf_model, tracker, stream_tracker, track_color_cache

    # Select tracker based on context
    active_tracker = stream_tracker if use_stream_tracker else tracker

    if hisdf_model is None:
        return image, "Error: Model not initialized. Please check model paths.", 0.0

    if image is None:
        return None, "No image provided", 0.0

    # Update model thresholds
    hisdf_model._obj_class_score_th = obj_threshold
    hisdf_model._attr_class_score_th = attr_threshold
    hisdf_model._keypoint_th = keypoint_threshold

    # Make a copy for processing
    debug_image = copy.deepcopy(image)
    debug_image_h, debug_image_w = debug_image.shape[:2]

    # Start timing
    start_time = time.time()

    # Run detection
    boxes, depth_map, _seg_map = hisdf_model(
        image=debug_image,
        disable_generation_identification_mode=not enable_generation,
        disable_gender_identification_mode=not enable_gender,
        disable_left_and_right_hand_identification_mode=not enable_hand_lr,
        disable_headpose_identification_mode=not enable_headpose,
    )

    # Depth map overlay
    if enable_depth_map_overlay:
        # Resize depth_map to match image size if needed
        if depth_map.shape[:2] != (debug_image_h, debug_image_w):
            depth_map_resized = cv2.resize(depth_map, (debug_image_w, debug_image_h), interpolation=cv2.INTER_LINEAR)
        else:
            depth_map_resized = depth_map

        # depth_map is already 0-255 uint8 from HISDF._postprocess
        depth_colormap = cv2.applyColorMap(depth_map_resized, cv2.COLORMAP_JET)

        # Blend depth colormap with original image
        alpha = 0.6
        debug_image = cv2.addWeighted(debug_image, 1.0 - alpha, depth_colormap, alpha, 0.0)

    # Instance segmentation overlay
    if enable_instance_segmentation_overlay:
        body_mask_count = 0

        for box in boxes:
            if box.classid != 0:
                continue
            if box.mask is None or not isinstance(box.mask, np.ndarray) or box.mask.size == 0:
                continue

            x1 = max(0, box.x1)
            y1 = max(0, box.y1)
            x2 = min(debug_image_w, box.x2)
            y2 = min(debug_image_h, box.y2)
            if x2 <= x1 or y2 <= y1:
                continue

            roi_w = x2 - x1
            roi_h = y2 - y1

            resized_mask = cv2.resize(
                box.mask.astype(np.float32),
                (roi_w, roi_h),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_binary = resized_mask >= 0.5
            if not np.any(mask_binary):
                continue

            body_mask_count += 1
            # Use track_id if available, otherwise use body_mask_count
            color_key = box.track_id if (enable_tracking and box.track_id > 0) else body_mask_count
            if color_key not in track_color_cache:
                track_color_cache[color_key] = np.array(get_id_color(color_key), dtype=np.float32)
            color = track_color_cache[color_key]

            alpha = 0.6

            roi = debug_image[y1:y2, x1:x2]
            blended_pixels = (
                roi[mask_binary].astype(np.float32) * (1.0 - alpha)
                + color * alpha
            )
            roi[mask_binary] = blended_pixels.astype(np.uint8)

    # Apply tracking
    if enable_tracking:
        body_boxes = [box for box in boxes if box.classid == 0]
        active_tracker.update(body_boxes)

        # Clean up track_color_cache for inactive tracks
        active_track_ids = {track['id'] for track in active_tracker.tracks}
        stale_ids = [tid for tid in list(track_color_cache.keys()) if tid not in active_track_ids and tid > 0]
        for tid in stale_ids:
            track_color_cache.pop(tid, None)

    # Draw bounding boxes and annotations
    white_line_width = 2
    colored_line_width = 1

    for box in boxes:
        classid = box.classid
        color = (255, 255, 255)

        # Determine color based on class and attributes
        if classid == 0:  # Body
            if enable_gender and box.gender == 0:
                color = (255, 0, 0)  # Male - Blue
            elif enable_gender and box.gender == 1:
                color = (139, 116, 225)  # Female - Purple
            else:
                color = (0, 200, 255)  # Unknown - Orange
        elif classid == 7:  # Head
            if enable_headpose and box.head_pose != -1:
                color = BOX_COLORS[box.head_pose][0]
            else:
                color = (0, 0, 255)
        elif classid == 16:  # Face
            color = (0, 200, 255)
            if enable_face_mosaic:
                w = int(abs(box.x2 - box.x1))
                h = int(abs(box.y2 - box.y1))
                if w > 0 and h > 0:
                    small_box = cv2.resize(
                        debug_image[box.y1 : box.y2, box.x1 : box.x2, :], (3, 3)
                    )
                    normal_box = cv2.resize(small_box, (w, h))
                    debug_image[box.y1 : box.y2, box.x1 : box.x2, :] = normal_box
        elif classid == 26:  # Hand
            if enable_hand_lr and box.handedness == 0:
                color = (0, 128, 0)  # Left hand
            elif enable_hand_lr and box.handedness == 1:
                color = (255, 0, 255)  # Right hand
            else:
                color = (0, 255, 0)
        elif classid in [21, 22, 23, 24, 25, 29, 30, 31, 32]:  # Keypoints
            if keypoint_mode in ["dot", "both"]:
                cv2.circle(debug_image, (box.cx, box.cy), 4, (255, 255, 255), -1)
                cv2.circle(debug_image, (box.cx, box.cy), 3, color, -1)
            if keypoint_mode in ["box", "both"]:
                cv2.rectangle(
                    debug_image, (box.x1, box.y1), (box.x2, box.y2), (255, 255, 255), 2
                )
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, 1)
            continue

        # Draw bounding box
        cv2.rectangle(
            debug_image,
            (box.x1, box.y1),
            (box.x2, box.y2),
            (255, 255, 255),
            white_line_width,
        )
        cv2.rectangle(
            debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width
        )

        # Draw attributes text
        if classid == 0:  # Body
            generation_txt = ""
            if enable_generation:
                if box.generation == 0:
                    generation_txt = "Adult"
                elif box.generation == 1:
                    generation_txt = "Child"

            gender_txt = ""
            if enable_gender:
                if box.gender == 0:
                    gender_txt = "M"
                elif box.gender == 1:
                    gender_txt = "F"

            attr_txt = (
                f"{generation_txt}({gender_txt})" if gender_txt else generation_txt
            )

            if attr_txt:
                cv2.putText(
                    debug_image,
                    attr_txt,
                    (box.x1, max(box.y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    debug_image,
                    attr_txt,
                    (box.x1, max(box.y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        elif classid == 7:  # Head
            if enable_headpose and box.head_pose != -1:
                headpose_txt = BOX_COLORS[box.head_pose][1]
                cv2.putText(
                    debug_image,
                    headpose_txt,
                    (box.x1, max(box.y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    debug_image,
                    headpose_txt,
                    (box.x1, max(box.y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            # Distance measurement
            if enable_distance:
                if camera_fov > 90:
                    focalLength = debug_image_w / (camera_fov * (math.pi / 180))
                else:
                    focalLength = debug_image_w / (
                        2 * math.tan((camera_fov / 2) * (math.pi / 180))
                    )
                distance = (AVERAGE_HEAD_WIDTH * focalLength) / abs(box.x2 - box.x1)

                cv2.putText(
                    debug_image,
                    f"{distance:.2f}m",
                    (box.x1 + 5, box.y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    debug_image,
                    f"{distance:.2f}m",
                    (box.x1 + 5, box.y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (10, 10, 10),
                    1,
                    cv2.LINE_AA,
                )

        elif classid == 26:  # Hand
            if enable_hand_lr:
                hand_txt = (
                    "L" if box.handedness == 0 else "R" if box.handedness == 1 else ""
                )
                if hand_txt:
                    cv2.putText(
                        debug_image,
                        hand_txt,
                        (box.x1, max(box.y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        debug_image,
                        hand_txt,
                        (box.x1, max(box.y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

        # Tracking ID
        if enable_tracking and classid == 0 and box.track_id > 0:
            track_text = f"ID: {box.track_id}"
            cv2.putText(
                debug_image,
                track_text,
                (box.x1, max(box.y1 - 30, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_image,
                track_text,
                (box.x1, max(box.y1 - 30, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                1,
                cv2.LINE_AA,
            )

    # Draw skeleton
    if enable_skeleton:
        draw_skeleton(
            image=debug_image, boxes=boxes, color=(0, 255, 255)
        )

    # Calculate inference time
    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    # Display inference time on the bottom-left corner
    time_text = f"Inference: {inference_time:.1f}ms ({1000 / inference_time:.1f}fps)"
    cv2.putText(
        debug_image,
        time_text,
        (10, debug_image_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        debug_image,
        time_text,
        (10, debug_image_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Create info text
    info = f"Detected: {len([b for b in boxes if b.classid == 0])} bodies, "
    info += f"{len([b for b in boxes if b.classid == 7])} heads, "
    info += f"{len([b for b in boxes if b.classid == 26])} hands"

    return debug_image, info, inference_time


def process_image_stream(
    image: np.ndarray,
    obj_threshold: float = 0.35,
    attr_threshold: float = 0.70,
    keypoint_threshold: float = 0.30,
    enable_skeleton: bool = True,
    enable_depth_map_overlay: bool = False,
    enable_instance_segmentation_overlay: bool = False,
    enable_gender: bool = True,
    enable_generation: bool = True,
    enable_headpose: bool = True,
    enable_hand_lr: bool = True,
    enable_tracking: bool = False,
    enable_face_mosaic: bool = False,
    enable_distance: bool = True,
    keypoint_mode: str = "dot",
    camera_fov: int = 90,
) -> np.ndarray:
    """Process image for streaming (only processes latest frame, skips old frames)."""
    global stream_lock, stream_latest_frame, stream_processing, stream_latest_output

    if image is None:
        return stream_latest_output if stream_latest_output is not None else image

    # Always update the latest frame
    with stream_lock:
        stream_latest_frame = image.copy()

        # If already processing, return the previous output immediately
        if stream_processing:
            return stream_latest_output if stream_latest_output is not None else image

        # Mark as processing
        stream_processing = True
        frame_to_process = stream_latest_frame.copy()

    try:
        # Process the frame
        output_image, _, _ = process_image(
            frame_to_process,
            obj_threshold,
            attr_threshold,
            keypoint_threshold,
            enable_skeleton,
            enable_depth_map_overlay,
            enable_instance_segmentation_overlay,
            enable_gender,
            enable_generation,
            enable_headpose,
            enable_hand_lr,
            enable_tracking,
            enable_face_mosaic,
            enable_distance,
            keypoint_mode,
            camera_fov,
            use_stream_tracker=True,  # Use separate tracker for streaming
        )

        # Update the latest output
        with stream_lock:
            stream_latest_output = output_image

        return output_image

    finally:
        # Mark as not processing
        with stream_lock:
            stream_processing = False


def process_video(
    video_path: str,
    obj_threshold: float = 0.35,
    attr_threshold: float = 0.70,
    keypoint_threshold: float = 0.30,
    enable_skeleton: bool = True,
    enable_depth_map_overlay: bool = False,
    enable_instance_segmentation_overlay: bool = False,
    enable_gender: bool = True,
    enable_generation: bool = True,
    enable_headpose: bool = True,
    enable_hand_lr: bool = True,
    enable_tracking: bool = True,
    enable_face_mosaic: bool = False,
    enable_distance: bool = True,
    keypoint_mode: str = "dot",
    camera_fov: int = 90,
    progress: gr.Progress = gr.Progress(),
) -> str:
    """Process video file frame by frame."""
    global tracker, track_color_cache

    if video_path is None:
        return None

    if hisdf_model is None:
        return None

    # Reset tracker and color cache for new video
    tracker = SimpleSortTracker()
    track_color_cache.clear()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: Could not open video file"

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output video
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    for frame_count in progress.tqdm(
        range(total_frames), desc=f"Processing video ({total_frames} frames)"
    ):
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed_frame, _, _ = process_image(
            frame,
            obj_threshold,
            attr_threshold,
            keypoint_threshold,
            enable_skeleton,
            enable_depth_map_overlay,
            enable_instance_segmentation_overlay,
            enable_gender,
            enable_generation,
            enable_headpose,
            enable_hand_lr,
            enable_tracking,
            enable_face_mosaic,
            enable_distance,
            keypoint_mode,
            camera_fov,
        )

        out.write(processed_frame)

    cap.release()
    out.release()

    return output_path


def create_gradio_interface():
    """Create Gradio interface."""

    # Scan for available models
    hisdf_models = scan_onnx_models(".")

    with gr.Blocks(title="473_HISDF") as demo:
        gr.Markdown("# 473_HISDF - Human Instance, Skeleton, and Depth Fusion")
        gr.Markdown(
            "Upload an image or use webcam for real-time detection of body parts, poses, depth estimation and instance segmentation."
        )

        with gr.Tabs():
            # Tab 1: Image Upload
            with gr.Tab("Image Upload"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            label="Input Image",
                            type="numpy",
                            sources=["upload", "clipboard"],
                        )

                        with gr.Accordion("Detection Settings", open=True):
                            obj_threshold = gr.Slider(
                                0.0,
                                1.0,
                                0.35,
                                label="Object Score Threshold",
                                step=0.05,
                            )
                            attr_threshold = gr.Slider(
                                0.0,
                                1.0,
                                0.70,
                                label="Attribute Score Threshold",
                                step=0.05,
                            )
                            keypoint_threshold = gr.Slider(
                                0.0,
                                1.0,
                                0.30,
                                label="Keypoint Score Threshold",
                                step=0.05,
                            )

                        with gr.Accordion("Visualization Settings", open=True):
                            enable_skeleton = gr.Checkbox(
                                True, label="Enable Skeleton Drawing"
                            )
                            overlay_group = gr.Radio(
                                ["none", "depth", "instance"],
                                value="instance",
                                label="Overlay Mode",
                            )
                            keypoint_mode = gr.Radio(
                                ["dot", "box", "both"],
                                value="dot",
                                label="Keypoint Drawing Mode",
                            )

                        with gr.Accordion("Attribute Settings", open=False):
                            enable_gender = gr.Checkbox(
                                True, label="Enable Gender Detection"
                            )
                            enable_generation = gr.Checkbox(
                                True, label="Enable Generation Detection (Adult/Child)"
                            )
                            enable_headpose = gr.Checkbox(
                                True, label="Enable Head Pose Detection"
                            )
                            enable_hand_lr = gr.Checkbox(
                                True, label="Enable Hand L/R Detection"
                            )

                        with gr.Accordion("Advanced Settings", open=False):
                            enable_tracking = gr.Checkbox(
                                False, label="Enable Tracking"
                            )
                            enable_face_mosaic = gr.Checkbox(
                                False, label="Enable Face Mosaic"
                            )
                            enable_distance = gr.Checkbox(
                                True, label="Enable Distance Measurement"
                            )
                            camera_fov = gr.Slider(
                                30, 180, 90, label="Camera Horizontal FOV", step=1
                            )

                        process_btn = gr.Button("Process Image", variant="primary")

                    with gr.Column(scale=1):
                        output_image = gr.Image(label="Output Image")

                # Example images
                gr.Examples(
                    examples=[],
                    inputs=input_image,
                    label="Example Images (Add your own)",
                )

                # Process button click
                def process_with_overlay(img, obj_th, attr_th, kp_th, skel, overlay, gender, gen, headpose, hand, track, mosaic, dist, kp_mode, fov):
                    enable_depth = (overlay == "depth")
                    enable_instance = (overlay == "instance")
                    return process_image(
                        img, obj_th, attr_th, kp_th, skel, enable_depth, enable_instance,
                        gender, gen, headpose, hand, track, mosaic, dist, kp_mode, fov
                    )[0]

                process_btn.click(
                    fn=process_with_overlay,
                    inputs=[
                        input_image,
                        obj_threshold,
                        attr_threshold,
                        keypoint_threshold,
                        enable_skeleton,
                        overlay_group,
                        enable_gender,
                        enable_generation,
                        enable_headpose,
                        enable_hand_lr,
                        enable_tracking,
                        enable_face_mosaic,
                        enable_distance,
                        keypoint_mode,
                        camera_fov,
                    ],
                    outputs=[output_image],
                )

                # Auto-process on image upload
                input_image.change(
                    fn=process_with_overlay,
                    inputs=[
                        input_image,
                        obj_threshold,
                        attr_threshold,
                        keypoint_threshold,
                        enable_skeleton,
                        overlay_group,
                        enable_gender,
                        enable_generation,
                        enable_headpose,
                        enable_hand_lr,
                        enable_tracking,
                        enable_face_mosaic,
                        enable_distance,
                        keypoint_mode,
                        camera_fov,
                    ],
                    outputs=[output_image],
                )

            # Tab 2: Video Upload
            with gr.Tab("Video Upload"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(
                            label="Input Video",
                            sources=["upload"],
                        )

                        with gr.Accordion("Detection Settings", open=True):
                            video_obj_threshold = gr.Slider(
                                0.0,
                                1.0,
                                0.35,
                                label="Object Score Threshold",
                                step=0.05,
                            )
                            video_attr_threshold = gr.Slider(
                                0.0,
                                1.0,
                                0.70,
                                label="Attribute Score Threshold",
                                step=0.05,
                            )
                            video_keypoint_threshold = gr.Slider(
                                0.0,
                                1.0,
                                0.30,
                                label="Keypoint Score Threshold",
                                step=0.05,
                            )

                        with gr.Accordion("Visualization Settings", open=True):
                            video_enable_skeleton = gr.Checkbox(
                                True, label="Enable Skeleton Drawing"
                            )
                            video_overlay_group = gr.Radio(
                                ["none", "depth", "instance"],
                                value="instance",
                                label="Overlay Mode",
                            )
                            video_keypoint_mode = gr.Radio(
                                ["dot", "box", "both"],
                                value="dot",
                                label="Keypoint Drawing Mode",
                            )

                        with gr.Accordion("Attribute Settings", open=False):
                            video_enable_gender = gr.Checkbox(
                                True, label="Enable Gender Detection"
                            )
                            video_enable_generation = gr.Checkbox(
                                True, label="Enable Generation Detection (Adult/Child)"
                            )
                            video_enable_headpose = gr.Checkbox(
                                True, label="Enable Head Pose Detection"
                            )
                            video_enable_hand_lr = gr.Checkbox(
                                True, label="Enable Hand L/R Detection"
                            )

                        with gr.Accordion("Advanced Settings", open=False):
                            video_enable_tracking = gr.Checkbox(
                                True, label="Enable Tracking"
                            )
                            video_enable_face_mosaic = gr.Checkbox(
                                False, label="Enable Face Mosaic"
                            )
                            video_enable_distance = gr.Checkbox(
                                True, label="Enable Distance Measurement"
                            )
                            video_camera_fov = gr.Slider(
                                30, 180, 90, label="Camera Horizontal FOV", step=1
                            )

                        process_video_btn = gr.Button(
                            "Process Video", variant="primary"
                        )

                    with gr.Column(scale=1):
                        video_output = gr.Video(label="Output Video")

                # Video processing event
                def process_video_with_overlay(vid, obj_th, attr_th, kp_th, skel, overlay, gender, gen, headpose, hand, track, mosaic, dist, kp_mode, fov, progress=gr.Progress()):
                    enable_depth = (overlay == "depth")
                    enable_instance = (overlay == "instance")
                    return process_video(
                        vid, obj_th, attr_th, kp_th, skel, enable_depth, enable_instance,
                        gender, gen, headpose, hand, track, mosaic, dist, kp_mode, fov, progress
                    )

                process_video_btn.click(
                    fn=process_video_with_overlay,
                    inputs=[
                        video_input,
                        video_obj_threshold,
                        video_attr_threshold,
                        video_keypoint_threshold,
                        video_enable_skeleton,
                        video_overlay_group,
                        video_enable_gender,
                        video_enable_generation,
                        video_enable_headpose,
                        video_enable_hand_lr,
                        video_enable_tracking,
                        video_enable_face_mosaic,
                        video_enable_distance,
                        video_keypoint_mode,
                        video_camera_fov,
                    ],
                    outputs=[video_output],
                )

            # Tab 3: Webcam Streaming
            with gr.Tab("Webcam Streaming"):
                with gr.Row():
                    with gr.Column(scale=1):
                        stream_input = gr.Image(
                            label="Input Image (Webcam)",
                            sources=["webcam"],
                            streaming=True,
                            type="numpy",
                            format="jpeg",
                        )

                        with gr.Accordion("Detection Settings", open=True):
                            stream_obj_threshold = gr.Slider(
                                0.0,
                                1.0,
                                0.35,
                                label="Object Score Threshold",
                                step=0.05,
                            )
                            stream_attr_threshold = gr.Slider(
                                0.0,
                                1.0,
                                0.70,
                                label="Attribute Score Threshold",
                                step=0.05,
                            )
                            stream_keypoint_threshold = gr.Slider(
                                0.0,
                                1.0,
                                0.30,
                                label="Keypoint Score Threshold",
                                step=0.05,
                            )

                        with gr.Accordion("Visualization Settings", open=True):
                            stream_enable_skeleton = gr.Checkbox(
                                True, label="Enable Skeleton Drawing"
                            )
                            stream_overlay_group = gr.Radio(
                                ["none", "depth", "instance"],
                                value="instance",
                                label="Overlay Mode",
                            )
                            stream_keypoint_mode = gr.Radio(
                                ["dot", "box", "both"],
                                value="dot",
                                label="Keypoint Drawing Mode",
                            )

                        with gr.Accordion("Attribute Settings", open=False):
                            stream_enable_gender = gr.Checkbox(
                                True, label="Enable Gender Detection"
                            )
                            stream_enable_generation = gr.Checkbox(
                                True, label="Enable Generation Detection (Adult/Child)"
                            )
                            stream_enable_headpose = gr.Checkbox(
                                True, label="Enable Head Pose Detection"
                            )
                            stream_enable_hand_lr = gr.Checkbox(
                                True, label="Enable Hand L/R Detection"
                            )

                        with gr.Accordion("Advanced Settings", open=False):
                            stream_enable_tracking = gr.Checkbox(
                                True, label="Enable Tracking"
                            )
                            stream_enable_face_mosaic = gr.Checkbox(
                                False, label="Enable Face Mosaic"
                            )
                            stream_enable_distance = gr.Checkbox(
                                True, label="Enable Distance Measurement"
                            )
                            stream_camera_fov = gr.Slider(
                                30, 180, 90, label="Camera Horizontal FOV", step=1
                            )

                    with gr.Column(scale=1):
                        stream_output = gr.Image(
                            label="Output Image (Detection Result)",
                            streaming=True,
                            type="numpy",
                            format="jpeg",
                        )

                # Real-time streaming processing
                def process_stream_with_overlay(img, obj_th, attr_th, kp_th, skel, overlay, gender, gen, headpose, hand, track, mosaic, dist, kp_mode, fov):
                    enable_depth = (overlay == "depth")
                    enable_instance = (overlay == "instance")
                    return process_image_stream(
                        img, obj_th, attr_th, kp_th, skel, enable_depth, enable_instance,
                        gender, gen, headpose, hand, track, mosaic, dist, kp_mode, fov
                    )

                stream_input.stream(
                    fn=process_stream_with_overlay,
                    inputs=[
                        stream_input,
                        stream_obj_threshold,
                        stream_attr_threshold,
                        stream_keypoint_threshold,
                        stream_enable_skeleton,
                        stream_overlay_group,
                        stream_enable_gender,
                        stream_enable_generation,
                        stream_enable_headpose,
                        stream_enable_hand_lr,
                        stream_enable_tracking,
                        stream_enable_face_mosaic,
                        stream_enable_distance,
                        stream_keypoint_mode,
                        stream_camera_fov,
                    ],
                    outputs=[stream_output],
                    time_limit=None,
                    show_progress="hidden",  # Hide progress completely
                )

            # Tab 4: Model Settings
            with gr.Tab("Model Settings"):
                gr.Markdown("## Select Model")
                gr.Markdown(
                    "Choose the ONNX model to use for detection, depth estimation and instance segmentation."
                )

                with gr.Row():
                    with gr.Column():
                        hisdf_dropdown = gr.Dropdown(
                            choices=hisdf_models,
                            value=hisdf_models[0] if hisdf_models else None,
                            label="HISDF Model (Detection + Depth + Instance Seg)",
                            interactive=True,
                        )

                        execution_provider = gr.Radio(
                            choices=["cpu", "cuda", "tensorrt"],
                            value="cuda",
                            label="Execution Provider",
                        )

                        load_models_btn = gr.Button(
                            "Load/Reload Model", variant="primary"
                        )

                    with gr.Column():
                        model_status = gr.Textbox(
                            label="Model Status",
                            lines=6,
                            interactive=False,
                            value="Please select model and click 'Load/Reload Model'",
                        )

                # Model loading event
                def load_models_wrapper(hisdf_path, provider):
                    return initialize_models(
                        hisdf_path, provider, force_reload=True
                    )

                load_models_btn.click(
                    fn=load_models_wrapper,
                    inputs=[hisdf_dropdown, execution_provider],
                    outputs=[model_status],
                )

    return demo


def main():
    """Main function to launch the Gradio app."""
    # Scan for available models
    print("Scanning for ONNX models...")
    hisdf_models = scan_onnx_models(".")

    print(f"Found {len(hisdf_models)} HISDF models: {hisdf_models}")

    # Auto-initialize with first available model
    if hisdf_models:
        print(f"\nAuto-initializing with:")
        print(f"  HISDF: {hisdf_models[0]}")
        print(f"  Execution Provider: cuda")
        status = initialize_models(hisdf_models[0], "cuda")
        print(status)
    else:
        print("\nWarning: No models found. Please add ONNX models to the directory.")
        print("Expected filenames:")
        print("  - deimv2*_depthanythingv2_instanceseg_*.onnx")

    # Create and launch interface
    demo = create_gradio_interface()
    demo.launch(share=True, server_port=7860)


if __name__ == "__main__":
    main()
