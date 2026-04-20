"""
feeds/webcam_feed.py
Webcam capture module for CAVSS DMS pipeline.
Captures frames from laptop webcam in a consistent format.
"""

import cv2
import time
import logging
import threading
from collections import deque
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class WebcamFeed:
    """
    Captures frames from laptop webcam for the DMS pipeline.

    Args:
        source: Webcam device index (default 0).
        width: Capture width in pixels.
        height: Capture height in pixels.
        fps: Target frames per second.
        flip_horizontal: Mirror the feed for natural feel.
    """

    def __init__(
        self,
        source: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        flip_horizontal: bool = True,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_horizontal = flip_horizontal

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._thread: Optional[threading.Thread] = None

        # FPS tracking
        self._frame_times: deque = deque(maxlen=30)
        self._frame_count = 0

    def start(self) -> bool:
        """Open webcam and start background capture thread.

        Returns:
            True if successfully opened, False otherwise.
        """
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            logger.error(f"Cannot open webcam at index {self.source}")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Read actual resolution (driver may override)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Webcam opened: {actual_w}x{actual_h} @ {self.fps} FPS (source={self.source})")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="WebcamFeed")
        self._thread.start()
        return True

    def _capture_loop(self) -> None:
        """Background thread: continuously read frames."""
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Webcam read failed — retrying")
                time.sleep(0.01)
                continue

            if self.flip_horizontal:
                frame = cv2.flip(frame, 1)

            with self._lock:
                self._latest_frame = frame

            self._frame_times.append(time.time())
            self._frame_count += 1

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the most recent frame.

        Returns:
            (success, frame) — success is False if no frame available yet.
        """
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    @property
    def fps_actual(self) -> float:
        """Current measured FPS from the capture loop."""
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def frame_count(self) -> int:
        """Total frames captured since start."""
        return self._frame_count

    def stop(self) -> None:
        """Stop capture thread and release webcam."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        logger.info("WebcamFeed stopped")

    def __enter__(self) -> "WebcamFeed":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    feed = WebcamFeed(flip_horizontal=True)
    if not feed.start():
        print("Failed to open webcam.")
        raise SystemExit(1)

    print("Webcam open. Press 'q' to quit.")
    try:
        while True:
            ok, frame = feed.read()
            if not ok:
                time.sleep(0.01)
                continue
            cv2.putText(
                frame,
                f"FPS: {feed.fps_actual:.1f}  Frames: {feed.frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("WebcamFeed Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        feed.stop()
        cv2.destroyAllWindows()
