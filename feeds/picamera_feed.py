"""
feeds/picamera_feed.py
PiCamera2 capture module for CAVSS DMS pipeline on Raspberry Pi 5.
Uses the libcamera-based picamera2 library for efficient capture.
"""

import time
import logging
import threading
from collections import deque
from typing import Optional, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Try importing Picamera2, but don't fail immediately so the file can be inspected
# on non-Pi systems.
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False


class PiCameraFeed:
    """
    Captures frames from Raspberry Pi Camera (e.g., v1.3/OV5647) via Picamera2.
    Adheres to the same interface as WebcamFeed.

    Args:
        source: Camera index (0 or 1 for Pi 5 CSI ports).
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

        self._picam2 = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._thread: Optional[threading.Thread] = None

        # FPS tracking
        self._frame_times: deque = deque(maxlen=30)
        self._frame_count = 0

    def start(self) -> bool:
        """Initialize PiCamera2 and start background capture thread."""
        if not PICAMERA_AVAILABLE:
            logger.error("Picamera2 is not installed or not supported on this OS.")
            return False

        try:
            # Initialize camera on the specified CSI port
            self._picam2 = Picamera2(camera=self.source)
            
            # Configure the video stream format. BGR888 is native for OpenCV
            config = self._picam2.create_video_configuration(
                main={"size": (self.width, self.height), "format": "BGR888"}
            )
            # Adjust framerate if the sensor allows it
            config["main"]["framerate"] = self.fps
            
            self._picam2.configure(config)
            
            # Additional controls like flip can be configured here if needed natively,
            # but we will handle it in software for consistency with WebcamFeed.
            
            self._picam2.start()
            logger.info(f"PiCamera opened: {self.width}x{self.height} @ {self.fps} FPS (source={self.source})")
            
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="PiCameraFeed")
            self._thread.start()
            return True
            
        except Exception as e:
            logger.error(f"Failed to start PiCamera2 on port {self.source}: {e}")
            return False

    def _capture_loop(self) -> None:
        """Background thread: continuously read frames from the camera queue."""
        while self._running:
            try:
                # Capture an array from the 'main' stream configuration
                # This will block until a new frame is ready
                frame = self._picam2.capture_array("main")
                
                if self.flip_horizontal:
                    frame = cv2.flip(frame, 1)

                with self._lock:
                    self._latest_frame = frame

                self._frame_times.append(time.time())
                self._frame_count += 1
                
            except Exception as e:
                logger.warning(f"PiCamera read failed: {e}")
                time.sleep(0.01)

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
        """Stop capture thread and release camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._picam2:
            self._picam2.stop()
            self._picam2.close()
        logger.info("PiCameraFeed stopped")

    def __enter__(self) -> "PiCameraFeed":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
