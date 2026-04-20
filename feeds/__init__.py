"""feeds — input source modules for CAVSS."""
from .webcam_feed import WebcamFeed
from .youtube_feed import YouTubeFeed, LocalVideoFeed

__all__ = ["WebcamFeed", "YouTubeFeed", "LocalVideoFeed"]
