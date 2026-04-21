# CAVSS — Context-Aware Vehicle Safety System

> A real-time AI driver safety system that doesn't just detect drowsiness — it **contextualizes risk** based on where, when, and how you're driving.

---

## Overview

CAVSS is a multi-modal driver safety platform built as a capstone project at **MIT-WPU, Pune** (Team Code: `25266P4-73`). It fuses a Driver Monitoring System (DMS), Advanced Driver Assistance (ADAS), and a custom **Context Risk Engine (CRE)** to compute a continuous, weighted safety score that adapts to real-world driving conditions.

The same drowsiness reading can produce a gentle reminder or a full alarm — depending on whether you're on a school zone at 3 AM in fog, or cruising a clear highway at noon.

---

## Architecture

```
CAVSS/
├── main.py                  # Orchestrator — runs all pipelines
├── config.yaml              # All thresholds, weights, and settings
├── requirements.txt
│
├── dms/                     # Driver Monitoring System
│   ├── drowsiness.py        # EAR + PERCLOS detection
│   ├── attention.py         # Head pose estimation
│   └── face_mesh.py         # MediaPipe 468-landmark face mesh
│
├── adas/                    # Advanced Driver Assistance
│   ├── lane_detection.py    # Lane departure warning
│   ├── object_detection.py  # YOLOv8-nano inference
│   └── collision_warning.py # Time-to-collision calculation
│
├── context_engine/          # Context Risk Engine (core innovation)
│   ├── risk_calculator.py   # Weighted fusion algorithm
│   ├── zone_manager.py      # Blackspot / school zone detection
│   ├── time_context.py      # Time-of-day risk multipliers
│   └── weather_context.py   # Visibility scoring from frame analysis
│
├── feeds/                   # Input sources
│   ├── youtube_feed.py      # Road dashcam video via yt-dlp
│   ├── webcam_feed.py       # Live driver face feed
│   └── mock_gps.py          # Simulated GPS route (demo)
│
├── alerts/                  # Alert output system
│   ├── audio_alert.py       # Graduated audio warnings
│   ├── visual_alert.py      # Dashboard indicators
│   └── alert_manager.py     # Alert escalation logic
│
├── interface/
│   └── dashboard.py         # Real-time Streamlit dashboard
│
├── data/
│   ├── blackspots.json      # Indian road blackspot coordinates
│   └── thresholds.json      # Calibrated detection thresholds
│
└── tests/
    └── test_cre.py          # CRE algorithm unit tests
```

---

## Core Innovation: Context Risk Engine (CRE)

Most drowsiness detectors are binary — awake or asleep. CAVSS computes a continuous **Risk Score** using a weighted, context-multiplied fusion:

```
Risk_Score = Σ(Wᵢ × Cᵢ × Sᵢ)
```

| Symbol | Meaning |
|--------|---------|
| `Wᵢ` | Base weight for parameter `i` |
| `Cᵢ` | Context multiplier (time, zone, visibility) |
| `Sᵢ` | Normalized sensor reading (0–1) |

### Base Weights

| Parameter | Weight |
|-----------|--------|
| Drowsiness (EAR + PERCLOS) | 0.30 |
| Attention (Head Pose) | 0.20 |
| Forward Collision (TTC) | 0.15 |
| Speed | 0.15 |
| Lane Position | 0.10 |
| Visibility | 0.10 |

### Context Multipliers

**Time of Day**

| Time | Multiplier |
|------|-----------|
| 00:00 – 05:00 | 1.5× (peak drowsiness) |
| 05:00 – 07:00 | 1.2× |
| 07:00 – 22:00 | 1.0× (baseline) |
| 22:00 – 00:00 | 1.3× |

**Zone Type**

| Zone | Multiplier |
|------|-----------|
| Blackspot | 1.5× |
| School Zone | 1.4× |
| Highway | 1.3× |
| Rural | 1.2× |
| Urban | 1.0× |

**Visibility**

| Condition | Multiplier |
|-----------|-----------|
| Fog | 1.4× |
| Rain | 1.3× |
| Night | 1.3× |
| Overcast | 1.1× |
| Clear | 1.0× |

### Alert Escalation

| Risk Score | Level | Response |
|------------|-------|----------|
| 0.0 – 0.3 | 🟢 Normal | No alert |
| 0.3 – 0.5 | 🟡 Caution | Gentle audio reminder |
| 0.5 – 0.7 | 🟠 Warning | Repeated beeps + visual |
| 0.7 – 0.85 | 🔴 Danger | Loud alarm + voice warning |
| 0.85 – 1.0 | ⛔ Critical | Continuous alarm + brake prep |

---

## Modules

### DMS — Driver Monitoring System

Uses **MediaPipe Face Mesh** (468 landmarks, 30+ FPS on CPU) for:

- **EAR (Eye Aspect Ratio)**: Triggers drowsiness if `EAR < 0.25` for 3+ consecutive frames
  ```
  EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
  ```
- **PERCLOS**: 1-minute rolling window; `> 40%` eye closure = severe drowsiness
- **Head Pose**: Pitch `> 20°` (nodding) or Yaw `> 30°` (looking away) for 2+ seconds = distraction

### ADAS — Advanced Driver Assistance

- **Lane Detection**: Canny edge → Hough transform with ROI masking; departure triggered at `>15%` centroid offset
- **Object Detection**: YOLOv8-nano (3.2M params, ~30 FPS on CPU) — detects cars, trucks, buses, motorcycles, bicycles, pedestrians at confidence `> 0.5`
- **Collision Warning**: TTC-based — caution at `< 3s`, critical at `< 1.5s`
  ```
  TTC = Distance / Relative_Velocity
  ```

### Feeds

Dual-feed architecture designed for laptop demo:
- **Webcam** → DMS pipeline (driver's face)
- **YouTube dashcam video** → ADAS pipeline (road scene)
- **Mock GPS** → simulated Pune route with blackspot zones

---

## Installation

```bash
git clone https://github.com/Shaunak0905/CAVSS.git
cd CAVSS
pip install -r requirements.txt

# Download YOLOv8-nano weights
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Dependencies

```
opencv-python>=4.8.0
mediapipe>=0.10.0
ultralytics>=8.0.0
numpy>=1.24.0
PyYAML>=6.0
yt-dlp>=2024.0.0
Pillow>=10.0.0
pygame>=2.5.0
pyttsx3>=2.90
```

---

## Usage

```bash
# Run full system
python main.py

# Run with a specific YouTube dashcam URL
python main.py --youtube-url "VIDEO_URL_HERE"

# Test individual modules
python -m dms.drowsiness --test
python -m adas.object_detection --test
python -m context_engine.risk_calculator --test
```

All thresholds, weights, and config values live in `config.yaml` — nothing is hardcoded.

---

## Demo Scenarios

| Scenario | Expected Behaviour |
|----------|--------------------|
| Normal driving, clear day | Low risk score, no alert |
| Drowsy driver in city | Yellow/orange alert |
| Drowsy driver, highway, 2 AM, near blackspot | Critical red alert |
| Lane departure | Visual + audio lane warning |
| Vehicle closing fast | Forward collision warning |

---

## Known Limitations

- YouTube feed has ~2–3 second latency
- GPS is simulated (no real GPS module)
- Weather detection is image-based approximation
- Blackspot dataset covers sample coordinates, not all of India
- No live vehicle integration — standalone simulation

---

## Roadmap

- [ ] Raspberry Pi 5 deployment
- [ ] Real GPS module (UART)
- [ ] IR camera for night-time DMS
- [ ] OBD-II integration for actual vehicle speed
- [ ] TensorRT optimization for edge inference
- [ ] Cloud dashboard for fleet management

---

## Project Info

| Field | Details |
|-------|---------|
| Institution | MIT-WPU, Pune |
| Team Code | 25266P4-73 |
| Subject | PBL-4 / Capstone |
| Semester | 6th (Pre-final year) |
| Language | Python 100% |

---

## License

This project was developed as an academic capstone. Contact the team for reuse or collaboration.
