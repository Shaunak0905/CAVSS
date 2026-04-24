"""
main.py
CAVSS — Context-Aware Vehicle Safety System
Orchestrator: runs DMS + ADAS + CRE + Alerts + Dashboard simultaneously.

Usage:
    python main.py
    python main.py --youtube-url "https://youtu.be/..."
    python main.py --local-video path/to/video.mp4
    python main.py --no-adas          # DMS only (no road feed)

Keyboard controls:
    q  — quit
    p  — pause/resume
    s  — screenshot
    d  — toggle debug overlay
    r  — toggle recording
    1  — scenario: normal city
    2  — scenario: highway night
    3  — scenario: blackspot danger
    4  — scenario: school zone rain
    v  — cycle visibility
    t  — set time to day (14:00)
    n  — set time to night (02:00)
    z  — simulate drowsiness (toggle)
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional
import numpy as np
import yaml
import cv2

# ---------------------------------------------------------------------------
# Configure logging before importing modules (they use the root logger)
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/cavss.log", mode="a"),
    ],
)
logger = logging.getLogger("CAVSS")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from feeds.webcam_feed import WebcamFeed
try:
    from feeds.picamera_feed import PiCameraFeed
except ImportError:
    PiCameraFeed = None
from feeds.youtube_feed import YouTubeFeed, LocalVideoFeed

from dms.face_mesh import FaceMeshProcessor
from dms.drowsiness import DrowsinessDetector, DrowsinessState
from dms.attention import AttentionDetector, AttentionState

from adas.object_detection import ObjectDetector, Detection
from adas.lane_detection import LaneDetector, LaneResult
from adas.collision_warning import CollisionWarning, CollisionRisk

from context_engine.risk_calculator import (
    ContextRiskEngine,
    SensorReadings,
    ContextState,
    RiskOutput,
    ScenarioSimulator,
)
from context_engine.zone_manager import ZoneManager

from alerts.alert_manager import AlertManager
from interface.dashboard import Dashboard


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CAVSS — Context-Aware Vehicle Safety System")
    p.add_argument("--youtube-url", default="", help="YouTube dashcam URL for ADAS feed")
    p.add_argument("--local-video", default="", help="Local video file path for ADAS feed")
    p.add_argument("--no-adas", action="store_true", help="Disable ADAS (road) feed")
    p.add_argument("--config", default="config.yaml", help="Path to config file")
    p.add_argument("--record", action="store_true", help="Record demo session")
    p.add_argument("--debug", action="store_true", help="Enable debug overlay")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Adjust log level from config
    log_level = cfg.get("system", {}).get("log_level", "INFO")
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))

    logger.info("=" * 60)
    logger.info("CAVSS Starting — MIT-WPU Team 25266P4-73")
    logger.info(f"Demo date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Initialise feeds
    # -----------------------------------------------------------------------
    feeds_cfg = cfg.get("feeds", {})

    feed_type = feeds_cfg["dms"].get("type", "webcam")
    
    if feed_type == "picamera":
        if PiCameraFeed is None:
            logger.error("PiCameraFeed is not available. Falling back to WebcamFeed.")
            feed_type = "webcam"
        else:
            dms_feed = PiCameraFeed(
                source=feeds_cfg["dms"]["source"],
                width=feeds_cfg["dms"]["width"],
                height=feeds_cfg["dms"]["height"],
                fps=feeds_cfg["dms"]["fps"],
                flip_horizontal=feeds_cfg["dms"]["flip_horizontal"],
            )
            
    if feed_type == "webcam":
        dms_feed = WebcamFeed(
            source=feeds_cfg["dms"]["source"],
            width=feeds_cfg["dms"]["width"],
            height=feeds_cfg["dms"]["height"],
            fps=feeds_cfg["dms"]["fps"],
            flip_horizontal=feeds_cfg["dms"]["flip_horizontal"],
        )

    adas_feed = None
    if not args.no_adas:
        youtube_url = args.youtube_url or feeds_cfg["adas"].get("youtube_url", "")
        local_path = args.local_video or feeds_cfg["adas"].get("local_path", "")
        adas_source = feeds_cfg["adas"]["source"]

        if local_path:
            adas_feed = LocalVideoFeed(
                path=local_path,
                loop=True,
                width=feeds_cfg["adas"]["width"],
                height=feeds_cfg["adas"]["height"],
            )
        elif adas_source == "youtube" or youtube_url:
            adas_feed = YouTubeFeed(
                youtube_url=youtube_url,
                width=feeds_cfg["adas"]["width"],
                height=feeds_cfg["adas"]["height"],
            )

    # -----------------------------------------------------------------------
    # Initialise DMS modules
    # -----------------------------------------------------------------------
    dms_cfg = cfg.get("dms", {})
    face_mesh = FaceMeshProcessor(
        max_num_faces=dms_cfg["face_mesh"]["max_num_faces"],
        refine_landmarks=dms_cfg["face_mesh"]["refine_landmarks"],
        min_detection_confidence=dms_cfg["face_mesh"]["min_detection_confidence"],
        min_tracking_confidence=dms_cfg["face_mesh"]["min_tracking_confidence"],
    )
    drowsiness_det = DrowsinessDetector(
        config=dms_cfg["drowsiness"],
        fps=feeds_cfg["dms"]["fps"],
    )
    attention_det = AttentionDetector(
        config=dms_cfg["attention"],
        frame_shape=(feeds_cfg["dms"]["height"], feeds_cfg["dms"]["width"]),
    )

    # -----------------------------------------------------------------------
    # Initialise ADAS modules
    # -----------------------------------------------------------------------
    adas_cfg = cfg.get("adas", {})
    object_det = ObjectDetector(
        model_path=adas_cfg["object_detection"]["model_path"],
        confidence_threshold=adas_cfg["object_detection"]["confidence_threshold"],
        iou_threshold=adas_cfg["object_detection"]["iou_threshold"],
        device=adas_cfg["object_detection"]["device"],
        target_classes=adas_cfg["object_detection"]["target_classes"],
    )
    lane_det = LaneDetector(
        config=adas_cfg["lane_detection"],
        frame_shape=(feeds_cfg["adas"]["height"], feeds_cfg["adas"]["width"]),
    )
    collision_warn = CollisionWarning(
        config=adas_cfg["collision"],
        frame_fps=cfg["system"]["target_fps"],
        frame_width=feeds_cfg["adas"]["width"],
    )

    # -----------------------------------------------------------------------
    # Initialise CRE, scenario simulator, zone manager
    # -----------------------------------------------------------------------
    zone_mgr = ZoneManager(blackspot_file="data/blackspots.json")
    cre = ContextRiskEngine(config_path=args.config, zone_manager=zone_mgr)
    scenario_sim = ScenarioSimulator()

    # -----------------------------------------------------------------------
    # Alerts and Dashboard
    # -----------------------------------------------------------------------
    alert_mgr = AlertManager(cfg)
    record_cfg = cfg.get("output", {}).get("recording", {})
    should_record = args.record or record_cfg.get("enabled", False)
    dashboard = Dashboard(
        record=should_record,
        output_path=f"output/recordings/demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
        fps=record_cfg.get("fps", 15),
    )

    # -----------------------------------------------------------------------
    # Start feeds
    # -----------------------------------------------------------------------
    if not dms_feed.start():
        logger.error("Failed to open webcam. Check connection and try again.")
        sys.exit(1)
    logger.info("Webcam (DMS) started")

    if adas_feed:
        if not adas_feed.start():
            logger.warning("ADAS feed failed to start — continuing DMS-only")
            adas_feed = None
        else:
            logger.info("ADAS feed started")

    # -----------------------------------------------------------------------
    # State
    # -----------------------------------------------------------------------
    paused = False
    show_debug = args.debug or cfg["system"].get("show_debug_overlay", True)
    simulate_drowsy = False
    frame_skip = cfg["system"].get("process_every_n_frames", 2)
    frame_num = 0
    screenshot_dir = cfg.get("output", {}).get("screenshots", {}).get("output_dir", "output/screenshots")

    # Last known values (used when frame skipped)
    dms_state = DrowsinessState()
    att_state = AttentionState()
    dms_frame_annotated: Optional[np.ndarray] = None
    adas_frame_annotated: Optional[np.ndarray] = None
    risk_output = RiskOutput(final_risk_score=0.0, alert_level="green")
    context: ContextState = scenario_sim.get_context()

    # FPS tracking
    loop_times: list = []
    max_fps_samples = 30

    logger.info("CAVSS running. Press 'q' to quit.")
    logger.info("Keys: 1=normal, 2=highway night, 3=blackspot, 4=school rain")
    logger.info("      v=cycle visibility, t=day, n=night, z=simulate drowsy")

    try:
        while True:
            t_loop_start = time.perf_counter()

            if paused:
                time.sleep(0.05)
                if adas_frame_annotated is not None or dms_frame_annotated is not None:
                    canvas = dashboard.render(
                        dms_frame_annotated, adas_frame_annotated,
                        risk_output, context, fps=0.0, show_debug=show_debug,
                    )
                    key = dashboard.show(canvas)
                    if _handle_key(key, scenario_sim, dashboard, canvas, screenshot_dir,
                                   locals()):
                        break
                continue

            # ----------------------------------------------------------------
            # Read DMS frame
            # ----------------------------------------------------------------
            dms_ok, dms_raw = dms_feed.read()
            if not dms_ok:
                time.sleep(0.005)
                continue

            # ----------------------------------------------------------------
            # Process DMS (every frame)
            # ----------------------------------------------------------------
            landmarks = face_mesh.process(dms_raw)
            face_detected = landmarks is not None

            if face_detected:
                left_eye, right_eye = face_mesh.get_eye_landmarks(landmarks)
                mouth = face_mesh.get_mouth_landmarks(landmarks)
                head_pts = face_mesh.get_head_pose_landmarks(landmarks)

                dms_state = drowsiness_det.update(left_eye, right_eye, mouth)
                att_state = attention_det.update(head_pts)

                if simulate_drowsy:
                    dms_state.drowsiness_score = min(1.0, dms_state.drowsiness_score + 0.5)
                    dms_state.is_eyes_closed = True
            else:
                drowsiness_det.reset()
                attention_det.reset()

            dms_frame_annotated = alert_mgr.annotate_dms(
                dms_raw,
                ear=dms_state.ear_avg,
                perclos=dms_state.perclos,
                pitch=att_state.pitch,
                yaw=att_state.yaw,
                drowsy=dms_state.is_eyes_closed,
                distracted=att_state.is_distracted,
                yawning=dms_state.is_yawning,
                face_detected=face_detected,
            )

            # ----------------------------------------------------------------
            # Process ADAS (every N frames to save CPU)
            # ----------------------------------------------------------------
            frame_num += 1
            adas_raw: Optional[np.ndarray] = None
            lane_result = LaneResult(None, None, 640, 0.0, False, 0.0)
            collision_result = CollisionRisk(None, "safe", None, 0.0)
            vis_condition = "clear"

            if adas_feed:
                adas_ok, adas_raw = adas_feed.read()
                if adas_ok and adas_raw is not None and (frame_num % frame_skip == 0):
                    detections, _ = object_det.detect(adas_raw)
                    lane_result = lane_det.detect(adas_raw)
                    collision_result = collision_warn.assess_risk(detections)

                    vis_condition, _ = _estimate_visibility(adas_raw)
                    vehicle_count = sum(1 for d in detections if d.class_name != "Person")
                    ped_count = sum(1 for d in detections if d.class_name == "Person")

                    # Annotate ADAS frame
                    adas_annotated = object_det.draw_detections(adas_raw, detections)
                    adas_annotated = lane_det.draw_lanes(adas_annotated, lane_result)
                    adas_annotated = alert_mgr.annotate_adas(
                        adas_annotated,
                        fps=adas_feed.fps_actual if hasattr(adas_feed, "fps_actual") else 0.0,
                        vehicle_count=vehicle_count,
                        pedestrian_count=ped_count,
                        lane_offset=lane_result.vehicle_offset,
                        ttc=collision_result.ttc,
                        visibility=vis_condition,
                    )
                    adas_frame_annotated = adas_annotated
                elif adas_raw is not None and adas_frame_annotated is None:
                    adas_frame_annotated = adas_raw
            

            # ----------------------------------------------------------------
            # CRE: build sensor readings and context, calculate risk
            # ----------------------------------------------------------------
            sensors = SensorReadings(
                drowsiness=dms_state.drowsiness_score,
                attention=att_state.attention_score,
                lane_departure=lane_result.departure_risk if adas_feed else 0.0,
                forward_collision=collision_result.forward_risk_score if adas_feed else 0.0,
            )

            # Update context visibility from ADAS analysis if we have a feed
            if adas_feed and adas_raw is not None:
                base_ctx = scenario_sim.get_context(
                    base_speed=cfg["simulation"]["gps"]["speed_kmh"]
                )
                context = ContextState(
                    time_of_day=base_ctx.time_of_day,
                    zone_type=base_ctx.zone_type,
                    visibility=vis_condition,
                    speed_kmh=base_ctx.speed_kmh,
                )
            else:
                context = scenario_sim.get_context()

            risk_output = cre.calculate_risk(sensors, context)

            # ----------------------------------------------------------------
            # Alerts
            # ----------------------------------------------------------------
            alert_mgr.process(risk_output)

            # Annotate DMS frame with risk overlay
            if dms_frame_annotated is not None:
                dms_frame_annotated = alert_mgr.annotate_risk(dms_frame_annotated, risk_output)

            # ----------------------------------------------------------------
            # FPS
            # ----------------------------------------------------------------
            loop_times.append(time.perf_counter())
            if len(loop_times) > max_fps_samples:
                loop_times.pop(0)
            elapsed_total = loop_times[-1] - loop_times[0] if len(loop_times) > 1 else 1.0
            system_fps = (len(loop_times) - 1) / elapsed_total if elapsed_total > 0 else 0.0

            # ----------------------------------------------------------------
            # Dashboard render
            # ----------------------------------------------------------------
            canvas = dashboard.render(
                dms_frame_annotated,
                adas_frame_annotated,
                risk_output,
                context,
                system_fps=system_fps,
                show_debug=show_debug,
            )
            key = dashboard.show(canvas)

            # ----------------------------------------------------------------
            # Keyboard handling (inline for simplicity)
            # ----------------------------------------------------------------
            if key == ord("q"):
                logger.info("Quit requested")
                break
            elif key == ord("p"):
                paused = not paused
                logger.info(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"{screenshot_dir}/screenshot_{ts}.png"
                os.makedirs(screenshot_dir, exist_ok=True)
                dashboard.screenshot(canvas, path)
            elif key == ord("d"):
                show_debug = not show_debug
            elif key == ord("r"):
                logger.info("Recording toggle not implemented in-loop — restart with --record")
            elif key == ord("1"):
                scenario_sim.switch("normal")
            elif key == ord("2"):
                scenario_sim.switch("highway_night")
            elif key == ord("3"):
                scenario_sim.switch("blackspot_danger")
            elif key == ord("4"):
                scenario_sim.switch("school_rain")
            elif key == ord("v"):
                new_vis = scenario_sim.cycle_visibility()
                logger.info(f"Visibility → {new_vis}")
            elif key == ord("t"):
                scenario_sim.set_day()
                logger.info("Time → 14:00 (Day)")
            elif key == ord("n"):
                scenario_sim.set_night()
                logger.info("Time → 02:00 (Night)")
            elif key == ord("z"):
                simulate_drowsy = not simulate_drowsy
                logger.info(f"Simulate drowsy: {simulate_drowsy}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
    finally:
        logger.info("Shutting down...")
        dms_feed.stop()
        if adas_feed:
            adas_feed.stop()
        face_mesh.close()
        alert_mgr.stop()
        dashboard.stop()
        logger.info("CAVSS stopped cleanly.")


def _handle_key(key: int, sim, dash, canvas, screenshot_dir, state: dict) -> bool:
    """Handle keyboard in paused state. Returns True if quit was pressed."""
    if key == ord("q"):
        return True
    if key == ord("p"):
        state["paused"] = False
    return False


def _estimate_visibility(frame: np.ndarray) -> tuple:
    """
    Quick image-based visibility estimation.

    Returns:
        (condition_str, multiplier_float)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    if brightness < 50:
        return "night", 1.3
    elif contrast < 25:
        return "fog", 1.4
    elif contrast < 35 and brightness < 120:
        return "overcast", 1.1
    elif contrast < 40:
        return "rain", 1.3
    return "clear", 1.0


if __name__ == "__main__":
    main()
