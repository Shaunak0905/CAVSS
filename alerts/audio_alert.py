"""
alerts/audio_alert.py
Graduated audio alerts via pygame (beeps) and pyttsx3 (voice).
Generates all sounds programmatically — no external audio files needed.
"""

import time
import logging
import threading
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


def _generate_beep_array(frequency: float, duration: float, volume: float = 0.5) -> np.ndarray:
    """
    Generate a stereo int16 sine wave suitable for pygame.sndarray.make_sound.

    Args:
        frequency: Tone frequency in Hz.
        duration: Duration in seconds.
        volume: Amplitude 0–1.

    Returns:
        (N, 2) int16 numpy array.
    """
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Fade in/out to avoid clicks
    fade_samples = min(int(sample_rate * 0.01), len(t) // 4)
    wave = np.sin(2 * np.pi * frequency * t) * volume
    if fade_samples > 0:
        wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
        wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    mono = (wave * 32767).astype(np.int16)
    return np.column_stack((mono, mono))


class AudioAlert:
    """
    Plays graduated audio alerts matched to CRE risk level.

    Levels and their sounds:
        green    → silent
        yellow   → gentle 440 Hz chime
        orange   → repeated 660 Hz beeps
        red      → loud 880 Hz alarm
        critical → rapid 1000 Hz siren

    Voice alerts fire on orange / red / critical via pyttsx3 (offline TTS).
    A cooldown per-channel prevents alert fatigue.

    Args:
        config: Dict from config.yaml['alerts'].
    """

    # Beep repeat counts by level
    _REPEAT: Dict[str, int] = {
        "yellow": 1,
        "orange": 2,
        "red": 3,
        "critical": 5,
    }

    def __init__(self, config: dict) -> None:
        self._enabled: bool = config.get("audio", {}).get("enabled", True)
        self._volume: float = config.get("audio", {}).get("volume", 0.7)
        voice_cfg = config.get("voice", {})
        self._voice_enabled: bool = voice_cfg.get("enabled", True)
        self._voice_rate: int = voice_cfg.get("rate", 150)
        self._cooldowns: Dict[str, float] = config.get("cooldown", {})

        self._pygame_ok = False
        self._tts_engine = None
        self._sounds: Dict[str, object] = {}
        self._last_played: Dict[str, float] = {}
        self._voice_lock = threading.Lock()
        self._voice_thread: Optional[threading.Thread] = None

        if self._enabled:
            self._init_pygame()

        if self._voice_enabled:
            self._init_tts()

        logger.info(
            f"AudioAlert ready (pygame={self._pygame_ok}, tts={self._tts_engine is not None})"
        )

    def _init_pygame(self) -> None:
        try:
            import pygame  # type: ignore
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()
            self._pygame = pygame
            self._pygame_ok = True

            freq_cfg = {
                "yellow":   440,
                "orange":   660,
                "red":      880,
                "critical": 1000,
            }
            dur_cfg = {
                "yellow":   0.2,
                "orange":   0.3,
                "red":      0.5,
                "critical": 0.1,
            }
            for level, freq in freq_cfg.items():
                arr = _generate_beep_array(freq, dur_cfg[level], self._volume)
                self._sounds[level] = pygame.sndarray.make_sound(arr)

        except ImportError:
            logger.warning("pygame not installed — audio alerts disabled")
        except Exception as exc:
            logger.warning(f"pygame init failed: {exc}")

    def _init_tts(self) -> None:
        try:
            import pyttsx3  # type: ignore
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty("rate", self._voice_rate)
            self._tts_engine.setProperty("volume", self._volume)
        except ImportError:
            logger.warning("pyttsx3 not installed — voice alerts disabled")
        except Exception as exc:
            logger.warning(f"pyttsx3 init failed: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def play(self, level: str) -> None:
        """
        Play a beep alert for the given risk level with cooldown enforcement.

        Args:
            level: One of "green", "yellow", "orange", "red", "critical".
        """
        if level == "green" or not self._pygame_ok:
            return

        cooldown = self._cooldowns.get("drowsiness", 10)
        now = time.time()
        last = self._last_played.get(f"beep_{level}", 0)
        if now - last < cooldown:
            return

        self._last_played[f"beep_{level}"] = now
        sound = self._sounds.get(level)
        if sound is None:
            return

        repeats = self._REPEAT.get(level, 1)
        try:
            for _ in range(repeats):
                sound.play()
                # Small gap between rapid beeps
                if level == "critical":
                    time.sleep(0.15)
        except Exception as exc:
            logger.debug(f"Beep play error: {exc}")

    def speak(self, message: str, level: str = "orange") -> None:
        """
        Speak a voice alert in a background thread (non-blocking).

        Args:
            message: Text to speak.
            level: Risk level — voice only fires for orange and above.
        """
        if not self._voice_enabled or self._tts_engine is None:
            return
        if level not in ("orange", "red", "critical"):
            return

        cooldown = self._cooldowns.get("voice", 5)
        now = time.time()
        if now - self._last_played.get("voice", 0) < cooldown:
            return
        self._last_played["voice"] = now

        def _speak() -> None:
            with self._voice_lock:
                try:
                    self._tts_engine.say(message)
                    self._tts_engine.runAndWait()
                except Exception as exc:
                    logger.debug(f"TTS speak error: {exc}")

        if self._voice_thread and self._voice_thread.is_alive():
            return  # Don't overlap voice alerts
        self._voice_thread = threading.Thread(target=_speak, daemon=True)
        self._voice_thread.start()

    def alert(self, level: str, message: Optional[str] = None) -> None:
        """
        Convenience method: play beep and optionally speak the message.

        Args:
            level: CRE alert level.
            message: Optional voice message text.
        """
        self.play(level)
        if message:
            self.speak(message, level)

    def stop(self) -> None:
        """Stop all audio immediately."""
        if self._pygame_ok:
            try:
                self._pygame.mixer.stop()
            except Exception:
                pass
