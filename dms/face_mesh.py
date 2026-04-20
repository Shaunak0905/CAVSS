"""
dms/face_mesh.py
MediaPipe Face Mesh wrapper — supports both legacy solutions API
(mediapipe < 0.10.14) and the new Tasks API (mediapipe 0.10.30+).
Auto-downloads the face_landmarker.task model when Tasks API is used.
"""

import logging
import os
import urllib.request
from typing import Optional, List, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Landmark index constants (identical across both APIs)
# ---------------------------------------------------------------------------
LEFT_EYE_INDICES:   List[int] = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES:  List[int] = [33, 160, 158, 133, 153, 144]
MOUTH_OUTER_INDICES: List[int] = [61, 291, 0, 17, 405, 321, 375, 78]
HEAD_POSE_INDICES:  List[int] = [1, 33, 263, 61, 291, 199]

# Tasks API model (auto-downloaded on first run, ~6 MB)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "face_landmarker.task",
)


# ---------------------------------------------------------------------------
# Internal backends
# ---------------------------------------------------------------------------

class _LegacyBackend:
    """Uses mp.solutions.face_mesh (mediapipe < 0.10.14)."""

    def __init__(self, max_num_faces, refine_landmarks,
                 min_detection_confidence, min_tracking_confidence):
        from mediapipe.solutions.face_mesh import FaceMesh  # type: ignore
        self._fm = FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._fm.process(rgb)
        if not results.multi_face_landmarks:
            return None
        lms = results.multi_face_landmarks[0].landmark
        return [(lm.x * w, lm.y * h, lm.z * w) for lm in lms]

    def close(self) -> None:
        self._fm.close()


class _TasksBackend:
    """Uses mediapipe.tasks FaceLandmarker (mediapipe 0.10.30+)."""

    def __init__(self, min_detection_confidence, min_tracking_confidence):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        # Download model if not present
        if not os.path.exists(_MODEL_PATH):
            os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
            logger.info("Downloading face_landmarker.task model (~6 MB) ...")
            try:
                urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
                logger.info("Model downloaded OK.")
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download FaceLandmarker model: {exc}\n"
                    f"Manually download from:\n{_MODEL_URL}\n"
                    f"and place it at: {_MODEL_PATH}"
                ) from exc

        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)
        self._mp = mp

    def process(self, frame: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )
        result = self._detector.detect(mp_image)
        if not result.face_landmarks:
            return None
        lms = result.face_landmarks[0]
        return [(lm.x * w, lm.y * h, lm.z * w) for lm in lms]

    def close(self) -> None:
        pass  # Tasks API handles its own cleanup


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class FaceMeshProcessor:
    """
    Thin wrapper around MediaPipe face landmark detection.
    Automatically selects the correct backend for the installed version.

    Args:
        max_num_faces: Maximum faces to track.
        refine_landmarks: Use iris refinement (legacy API only; Tasks model
                          always includes iris landmarks).
        min_detection_confidence: Detection threshold.
        min_tracking_confidence: Tracking threshold.
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        # Try legacy API first; fall back to Tasks API
        try:
            self._backend = _LegacyBackend(
                max_num_faces, refine_landmarks,
                min_detection_confidence, min_tracking_confidence,
            )
            logger.info("FaceMeshProcessor: using legacy solutions API")
        except (ImportError, AttributeError, ModuleNotFoundError):
            logger.info("FaceMeshProcessor: legacy API unavailable, switching to Tasks API")
            self._backend = _TasksBackend(
                min_detection_confidence, min_tracking_confidence
            )
            logger.info("FaceMeshProcessor: Tasks API ready")

    def process(
        self, frame: np.ndarray
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        Run face landmark detection on a BGR frame.

        Returns:
            List of (x_px, y_px, z_px) for each of the 468/478 landmarks,
            or None if no face is detected.
        """
        try:
            return self._backend.process(frame)
        except Exception as exc:
            logger.debug(f"FaceMesh process error: {exc}")
            return None

    def get_eye_landmarks(
        self, landmarks: List[Tuple[float, float, float]]
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        left  = [(landmarks[i][0], landmarks[i][1]) for i in LEFT_EYE_INDICES]
        right = [(landmarks[i][0], landmarks[i][1]) for i in RIGHT_EYE_INDICES]
        return left, right

    def get_mouth_landmarks(
        self, landmarks: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float]]:
        return [(landmarks[i][0], landmarks[i][1]) for i in MOUTH_OUTER_INDICES]

    def get_head_pose_landmarks(
        self, landmarks: List[Tuple[float, float, float]]
    ) -> np.ndarray:
        pts = [(landmarks[i][0], landmarks[i][1]) for i in HEAD_POSE_INDICES]
        return np.array(pts, dtype=np.float64)

    def close(self) -> None:
        self._backend.close()
        logger.info("FaceMeshProcessor closed")

    def __enter__(self) -> "FaceMeshProcessor":
        return self

    def __exit__(self, *_) -> None:
        self.close()
