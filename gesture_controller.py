"""
gesture_controller.py

GestureController: Uses Ultralytics YOLO-style hand-pose + face models (PyTorch) to
replace MediaPipe for the Nokia Snake gesture controls.

API:
    gc = GestureController(hand_model="path_or_model_name", face_model="path_or_model_name")
    direction, is_pinching, annotated = gc.detect_gestures(frame)
    gc.reset_gesture_state()

Notes:
- The code attempts to read 'keypoints' from various result attributes that community
  YOLO hand-pose checkpoints may use. If your checkpoint uses a different attribute,
  update the `_extract_keypoints_from_result()` method.
- If no keypoints are provided by the model, the code falls back to bounding-box
  midpoint tracking for swipe detection. Pinch detection requires landmarks; otherwise
  it's disabled (False).
"""

from typing import Tuple, Optional, List
import os
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:
    raise RuntimeError(
        "ultralytics package not found in the active Python environment.\n"
        "Install it with:\n  python -m pip install ultralytics\n"
        "Then make sure VS Code is using the same interpreter you installed into."
    ) from e
# ...existing code...

class GestureController:
    def __init__(
        self,
        hand_model: str = "yolov8n-handpose.pt",
        face_model: Optional[str] = None,
        device: str = "cpu",
        gesture_threshold: float = 0.05,
        max_cooldown: int = 10,
    ):
        """
        hand_model: path or model name for a hand-pose YOLO checkpoint (must produce keypoints ideally)
        face_model: optional model path/name for face detection (bounding boxes)
        device: 'cpu' or 'cuda'
        """
        self.device = device
        self.hand_model_path = hand_model
        self.face_model_path = face_model
        self.gesture_threshold = gesture_threshold
        self.max_cooldown = max_cooldown

        # load models
        self.hand_model = None
        self.face_model = None

        self._load_models()

        # gesture state
        self.previous_position = None
        self.current_direction = None
        self.gesture_cooldown = 0

    def _load_models(self):
        # Load hand model (may download if a model name is provided and ultralytics supports it)
        try:
            self.hand_model = YOLO(self.hand_model_path)
            # set device if supported
            try:
                self.hand_model.to(self.device)
            except Exception:
                pass
        except Exception as e:
            raise RuntimeError(
                f"Failed to load hand model '{self.hand_model_path}': {e}\n"
                "Provide a valid YOLO hand-pose model or path. Example community models exist "
                "named like 'yolov8n-handpose.pt' (user must download/provide)."
            )

        # Optional face model
        if self.face_model_path:
            try:
                self.face_model = YOLO(self.face_model_path)
                try:
                    self.face_model.to(self.device)
                except Exception:
                    pass
            except Exception as e:
                # Not fatal; just warn and continue without face detection
                print(f"Warning: could not load face model '{self.face_model_path}': {e}")
                self.face_model = None

    # --- Helpers to interpret model output (attempt robust extraction) ---
    def _extract_keypoints_from_result(self, res) -> Optional[np.ndarray]:
        """
        Attempt to find keypoints (NxKx2) in a result object from ultralytics model.
        Returns (num_persons_or_hands, num_keypoints, 2) numpy array or None.
        Different model checkpoints name keypoints differently; we try common options.
        """
        # 1) Common ultralytics pose models expose 'keypoints' attr on results
        if hasattr(res, "keypoints") and res.keypoints is not None:
            try:
                # convert to numpy array [N, K, 2]
                kps = np.array(res.keypoints.xy)  # may be [N, K, 2]
                return kps
            except Exception:
                pass

        # 2) Newer ultralytics versions return .masks, .boxes, .probs, and sometimes .keypoints
        # We try to access attributes that community checkpoints sometimes provide.
        if hasattr(res, "boxes") and res.boxes is not None:
            boxes = res.boxes
            # sometimes .keypoints stored under boxes.keypoints
            if hasattr(boxes, "keypoints") and boxes.keypoints is not None:
                try:
                    kps = np.array(boxes.keypoints.cpu())
                    return kps
                except Exception:
                    pass

        # 3) There is also a chance results.orig_keypoints or results.pose_keypoints exist
        for candidate in ["orig_keypoints", "pose_keypoints", "kpts", "landmarks"]:
            if hasattr(res, candidate):
                attr = getattr(res, candidate)
                if attr is not None:
                    try:
                        return np.array(attr)
                    except Exception:
                        pass

        # 4) If nothing found, return None (caller will fallback to bbox center)
        return None

    def _get_primary_hand_kps(self, keypoints_array: np.ndarray) -> Optional[np.ndarray]:
        """
        Given the keypoints_array returned by _extract_keypoints_from_result, pick the
        highest confidence or first hand and return its keypoints as (K,2).
        This function assumes the shape is either (H, K, 2) or (K, 2).
        """
        if keypoints_array is None:
            return None
        kps = keypoints_array
        if kps.ndim == 3 and kps.shape[0] >= 1:
            return kps[0]  # take first hand
        if kps.ndim == 2:
            return kps
        return None

    # --- Public API ---
    def detect_gestures(self, frame: np.ndarray) -> Tuple[Optional[str], bool, np.ndarray]:
        """
        Process a BGR frame (OpenCV) and return (direction, is_pinching, annotated_frame).

        direction: One of "UP", "DOWN", "LEFT", "RIGHT", or None
        is_pinching: True if pinch detected
        annotated_frame: BGR frame with drawings
        """
        annotated = frame.copy()
        height, width = frame.shape[:2]

        direction = None
        is_pinching = False

        # ---- Face detection (optional) ----
        if self.face_model:
            try:
                face_results = self.face_model.predict(source=frame, imgsz=640, conf=0.45, device=self.device)
                if face_results and len(face_results) > 0:
                    # face_results[0].boxes.xyxy may exist
                    res0 = face_results[0]
                    if hasattr(res0, "boxes") and res0.boxes is not None:
                        for box in res0.boxes:
                            try:
                                xyxy = box.xyxy.tolist()  # [x1,y1,x2,y2]
                            except Exception:
                                try:
                                    xyxy = box.xyxy.cpu().numpy().tolist()
                                except Exception:
                                    continue
                            x1, y1, x2, y2 = map(int, xyxy[:4])
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (200, 200, 0), 2)
                            cv2.putText(annotated, "Face", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
            except Exception as e:
                # Non-fatal; continue without face boxes
                print("Face detection error:", e)

        # ---- Hand detection / pose inference ----
        try:
            hand_results = self.hand_model.predict(source=frame, imgsz=640, conf=0.35, device=self.device)
        except Exception as e:
            raise RuntimeError(f"Error running hand model prediction: {e}\n"
                               "Check your hand_model checkpoint and ultralytics install.")

        # hand_results is list-like; get first result object
        if not hand_results or len(hand_results) == 0:
            # No detection; update cooldown and return
            if self.gesture_cooldown > 0:
                self.gesture_cooldown -= 1
            if self.current_direction:
                cv2.putText(annotated, f"Direction: {self.current_direction}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            return self.current_direction, False, annotated

        res0 = hand_results[0]

        # Try to extract hand keypoints (preferred)
        keypoints_all = self._extract_keypoints_from_result(res0)  # maybe None or [H,K,2]
        primary_kps = self._get_primary_hand_kps(keypoints_all)

        # If we have keypoints, try to map thumb/index landmarks
        # NOTE: Keypoint index mapping may differ between models. Common MediaPipe-like mapping:
        # 0: wrist
        # 1-4: thumb (4 = thumb_tip)
        # 5-8: index (8 = index_tip)
        # 12 = middle_tip, etc.
        # If your checkpoint uses a different ordering, change these indices.
        THUMB_TIP_IDX = 4
        INDEX_TIP_IDX = 8
        WRIST_IDX = 0

        current_pos = None

        if primary_kps is not None and primary_kps.shape[0] >= max(THUMB_TIP_IDX, INDEX_TIP_IDX, WRIST_IDX) + 1:
            # primary_kps are normalized pixel coordinates or absolute? Many UL models return pixel coords.
            # We attempt to detect by scale: if values <=1.0 assume normalized -> convert to pixel coords.
            kps = primary_kps.copy()
            if np.max(kps) <= 1.05:
                # normalized -> map to pixels
                kps[:, 0] = kps[:, 0] * width
                kps[:, 1] = kps[:, 1] * height
            # Extract wrist (or centroid) for swipe movement
            wrist = kps[WRIST_IDX]
            current_pos = np.array([wrist[0] / width, wrist[1] / height])  # normalized center for movement thresholding

            # Draw keypoints & connections (simple)
            for i, (x, y) in enumerate(kps[:, :2].astype(np.int32)):
                cv2.circle(annotated, (int(x), int(y)), 3, (0, 255, 0), -1)
                if i == THUMB_TIP_IDX:
                    cv2.putText(annotated, "T", (int(x) + 4, int(y) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if i == INDEX_TIP_IDX:
                    cv2.putText(annotated, "I", (int(x) + 4, int(y) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Pinch detection: distance between thumb tip and index tip
            thumb = kps[THUMB_TIP_IDX][:2] / np.array([width, height])
            index = kps[INDEX_TIP_IDX][:2] / np.array([width, height])
            distance = np.linalg.norm(thumb - index)
            is_pinching = distance < 0.05  # same threshold as your MediaPipe implementation
            if is_pinching:
                cv2.putText(annotated, "SPEED BOOST!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            # No usable keypoints found; fall back to bounding-box midpoint for motion
            is_pinching = False
            if hasattr(res0, "boxes") and res0.boxes is not None and len(res0.boxes) > 0:
                # take first bounding box
                try:
                    box = res0.boxes[0]
                    xyxy = None
                    try:
                        xyxy = box.xyxy.tolist()
                    except Exception:
                        try:
                            xyxy = box.xyxy.cpu().numpy().tolist()
                        except Exception:
                            pass
                    if xyxy:
                        x1, y1, x2, y2 = map(float, xyxy[:4])
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        current_pos = np.array([cx / width, cy / height])
                        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 160, 200), 2)
                        cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 160, 200), -1)
                except Exception:
                    current_pos = None

        # ---- Swipe gesture detection using current_pos and previous_position ----
        if current_pos is not None:
            if self.previous_position is not None and self.gesture_cooldown <= 0:
                movement = current_pos - self.previous_position
                if np.linalg.norm(movement) > self.gesture_threshold:
                    # horizontal vs vertical
                    if abs(movement[0]) > abs(movement[1]):
                        direction = "RIGHT" if movement[0] > 0 else "LEFT"
                    else:
                        # note: y increases downward in pixel coords, but since we normalized
                        # using screen coordinates, positive y means downward movement
                        direction = "DOWN" if movement[1] > 0 else "UP"

                    if direction != self.current_direction:
                        self.current_direction = direction
                        self.gesture_cooldown = self.max_cooldown
            # update previous
            self.previous_position = current_pos

        # cooldown decrement
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1

        # display current direction
        if self.current_direction:
            cv2.putText(annotated, f"Direction: {self.current_direction}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        return self.current_direction, is_pinching, annotated

    def reset_gesture_state(self):
        """Reset internal gesture state (useful between rounds)"""
        self.previous_position = None
        self.current_direction = None
        self.gesture_cooldown = 0
