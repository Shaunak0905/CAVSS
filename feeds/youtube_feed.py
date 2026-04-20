"""
feeds/youtube_feed.py
YouTube dashcam video feed for the CAVSS ADAS pipeline.
Uses yt-dlp to extract a direct stream URL, then reads with OpenCV.
"""

import cv2
import time
import logging
import threading
import subprocess
import shutil
from collections import deque
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Fallback test video URL (Big Buck Bunny — always available on YouTube)
FALLBACK_URL = "https://www.youtube.com/watch?v=aqz-KE-bpKQ"


def _get_stream_url(youtube_url: str, quality: str = "best[height<=720]") -> Optional[str]:
    """Extract a direct video stream URL using yt-dlp.

    Args:
        youtube_url: Full YouTube page URL.
        quality: yt-dlp format selector.

    Returns:
        Direct stream URL string, or None on failure.
    """
    if not shutil.which("yt-dlp"):
        logger.error("yt-dlp not found. Install with: pip install yt-dlp")
        return None

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--get-url",
                "-f", quality,
                "--no-playlist",
                youtube_url,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.error(f"yt-dlp error: {result.stderr.strip()}")
            return None
        stream_url = result.stdout.strip().split("\n")[0]
        if not stream_url:
            logger.error("yt-dlp returned empty URL")
            return None
        logger.info(f"Stream URL extracted (len={len(stream_url)})")
        return stream_url
    except subprocess.TimeoutExpired:
        logger.error("yt-dlp timed out after 30s")
        return None
    except Exception as exc:
        logger.error(f"Failed to get stream URL: {exc}")
        return None


class YouTubeFeed:
    """
    Streams frames from a YouTube video for the ADAS pipeline.

    Extracts a direct stream URL via yt-dlp and reads it with OpenCV.
    Runs a background thread to buffer the latest frame so the main
    processing loop never blocks on network I/O.

    Args:
        youtube_url: Full YouTube video URL.
        width: Resize width (0 = no resize).
        height: Resize height (0 = no resize).
        quality: yt-dlp quality selector.
    """

    def __init__(
        self,
        youtube_url: str = "",
        width: int = 1280,
        height: int = 720,
        quality: str = "best[height<=720]",
    ) -> None:
        self.youtube_url = youtube_url or FALLBACK_URL
        self.width = width
        self.height = height
        self.quality = quality

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._thread: Optional[threading.Thread] = None
        self._stream_url: Optional[str] = None

        # FPS tracking
        self._frame_times: deque = deque(maxlen=30)
        self._frame_count = 0
        self._reconnect_attempts = 0
        self._max_reconnects = 3

    def start(self) -> bool:
        """Resolve YouTube URL and start background capture thread.

        Returns:
            True if stream opened successfully, False otherwise.
        """
        logger.info(f"Resolving YouTube URL: {self.youtube_url}")
        self._stream_url = _get_stream_url(self.youtube_url, self.quality)
        if not self._stream_url:
            logger.warning("Falling back to FALLBACK_URL")
            self._stream_url = _get_stream_url(FALLBACK_URL, self.quality)
        if not self._stream_url:
            logger.error("Could not resolve any stream URL")
            return False

        if not self._open_capture():
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="YouTubeFeed"
        )
        self._thread.start()
        logger.info("YouTubeFeed started")
        return True

    def _open_capture(self) -> bool:
        """Open cv2.VideoCapture from resolved stream URL."""
        cap = cv2.VideoCapture(self._stream_url)
        # Give it a moment to negotiate
        time.sleep(0.5)
        if not cap.isOpened():
            logger.error("cv2.VideoCapture failed to open stream URL")
            return False
        self._cap = cap
        logger.info("VideoCapture opened for YouTube stream")
        return True

    def _capture_loop(self) -> None:
        """Background thread: continuously read frames, reconnect on failure."""
        consecutive_failures = 0
        max_failures = 30  # ~2s at 15 FPS before reconnect attempt

        while self._running:
            if self._cap is None:
                time.sleep(0.1)
                continue

            ret, frame = self._cap.read()

            if not ret:
                consecutive_failures += 1
                logger.warning(f"YouTube read failed (consecutive={consecutive_failures})")
                if consecutive_failures >= max_failures:
                    if self._reconnect_attempts < self._max_reconnects:
                        logger.info("Attempting stream reconnect...")
                        self._reconnect()
                        consecutive_failures = 0
                    else:
                        logger.error("Max reconnect attempts reached. Stopping YouTubeFeed.")
                        self._running = False
                        break
                time.sleep(0.05)
                continue

            consecutive_failures = 0

            # Resize if requested
            if self.width > 0 and self.height > 0:
                h, w = frame.shape[:2]
                if w != self.width or h != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

            with self._lock:
                self._latest_frame = frame

            self._frame_times.append(time.time())
            self._frame_count += 1

    def _reconnect(self) -> None:
        """Re-resolve URL and reopen capture."""
        self._reconnect_attempts += 1
        logger.info(f"Reconnect attempt {self._reconnect_attempts}/{self._max_reconnects}")
        if self._cap:
            self._cap.release()
            self._cap = None
        # Re-resolve stream URL (CDN URLs expire)
        new_url = _get_stream_url(self.youtube_url, self.quality)
        if new_url:
            self._stream_url = new_url
            self._open_capture()
        else:
            logger.warning("Reconnect: failed to get new stream URL")

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

    @property
    def is_alive(self) -> bool:
        """True if capture thread is still running."""
        return self._running and (self._thread is not None and self._thread.is_alive())

    def stop(self) -> None:
        """Stop capture thread and release resources."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
        logger.info("YouTubeFeed stopped")

    def __enter__(self) -> "YouTubeFeed":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()


class LocalVideoFeed:
    """
    Reads frames from a local video file — useful fallback when no internet.

    Args:
        path: Path to video file.
        loop: If True, restart from beginning when video ends.
        width: Resize width (0 = no resize).
        height: Resize height (0 = no resize).
    """

    def __init__(
        self,
        path: str,
        loop: bool = True,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.path = path
        self.loop = loop
        self.width = width
        self.height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._thread: Optional[threading.Thread] = None
        self._frame_times: deque = deque(maxlen=30)
        self._frame_count = 0

    def start(self) -> bool:
        """Open video file and start capture thread."""
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            logger.error(f"Cannot open video file: {self.path}")
            return False
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
        logger.info(f"LocalVideoFeed: {self.path} @ {fps:.1f} FPS")
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="LocalVideoFeed"
        )
        self._thread.start()
        return True

    def _capture_loop(self) -> None:
        video_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
        frame_delay = 1.0 / video_fps

        while self._running:
            t0 = time.time()
            ret, frame = self._cap.read()
            if not ret:
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self._running = False
                    break

            if self.width > 0 and self.height > 0:
                h, w = frame.shape[:2]
                if w != self.width or h != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

            with self._lock:
                self._latest_frame = frame

            self._frame_times.append(time.time())
            self._frame_count += 1

            # Maintain video playback speed
            elapsed = time.time() - t0
            sleep_t = frame_delay - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    @property
    def fps_actual(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()

    def __enter__(self) -> "LocalVideoFeed":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)

    url = sys.argv[1] if len(sys.argv) > 1 else FALLBACK_URL
    feed = YouTubeFeed(youtube_url=url, width=1280, height=720)

    if not feed.start():
        print("Failed to start YouTube feed.")
        raise SystemExit(1)

    print("YouTube feed open. Waiting for first frame... Press 'q' to quit.")
    # Wait up to 10s for first frame
    for _ in range(100):
        ok, frame = feed.read()
        if ok:
            break
        time.sleep(0.1)

    try:
        while True:
            ok, frame = feed.read()
            if not ok:
                time.sleep(0.01)
                continue
            cv2.putText(
                frame,
                f"YouTube FPS: {feed.fps_actual:.1f}  Frames: {feed.frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("YouTubeFeed Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        feed.stop()
        cv2.destroyAllWindows()
