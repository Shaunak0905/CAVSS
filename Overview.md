# CAVSS - Context-Aware Vehicle Safety System

## Project Identity
- **Team Code**: 25266P4-73
- **Institution**: MIT-WPU, Pune
- **Semester**: 6th (Pre-final year)
- **Subject**: PBL-4 / Capstone Project
- **Demo Date**: March 28, 2026

## Core Innovation: Context Risk Engine (CRE)

The CRE is what separates this from every other drowsiness detector. It doesn't just detect — it **contextualizes**.

The same drowsiness reading produces completely different responses based on:
- WHERE you're driving (city vs highway vs blackspot zone)
- WHEN you're driving (2 PM vs 2 AM)
- HOW FAST you're driving (20 km/h vs 100 km/h)
- WHAT'S VISIBLE (clear day vs fog vs rain)
- WHAT'S AROUND (pedestrians, vehicles, lane markings)

**Formula:**
```
Risk_Score = Σ(Wi × Ci × Si)

Where:
- Wi = Base weight for parameter i
- Ci = Context multiplier (time, zone, visibility)
- Si = Sensor reading (normalized 0-1)
```

## Architecture Overview

```
CAVSS/
├── main.py                 # Orchestrator - runs everything
├── config.yaml             # All thresholds, weights, API keys
├── requirements.txt        # Dependencies
│
├── dms/                    # Driver Monitoring System
│   ├── __init__.py
│   ├── drowsiness.py       # EAR + PERCLOS detection
│   ├── attention.py        # Head pose estimation
│   └── face_mesh.py        # MediaPipe face landmarks
│
├── adas/                   # Advanced Driver Assistance
│   ├── __init__.py
│   ├── lane_detection.py   # Lane departure warning
│   ├── object_detection.py # YOLOv8-nano inference
│   └── collision_warning.py # Time-to-collision calculation
│
├── context_engine/         # The CRE - Core Innovation
│   ├── __init__.py
│   ├── risk_calculator.py  # Weighted fusion algorithm
│   ├── zone_manager.py     # Blackspot/school zone detection
│   ├── time_context.py     # Time-of-day risk adjustment
│   └── weather_context.py  # Visibility scoring
│
├── feeds/                  # Input Sources
│   ├── __init__.py
│   ├── youtube_feed.py     # Road video from YouTube
│   ├── webcam_feed.py      # Driver face from webcam
│   └── mock_gps.py         # Simulated GPS for demo
│
├── alerts/                 # Output/Response System
│   ├── __init__.py
│   ├── audio_alert.py      # Graduated audio warnings
│   ├── visual_alert.py     # Dashboard indicators
│   └── alert_manager.py    # Alert escalation logic
│
├── interface/              # Demo Dashboard
│   ├── __init__.py
│   └── dashboard.py        # Real-time visualization
│
├── models/                 # Pre-trained models
│   └── README.md           # Download instructions
│
├── data/                   # Datasets and zone data
│   ├── blackspots.json     # Indian road blackspot coordinates
│   └── thresholds.json     # Calibrated detection thresholds
│
└── tests/                  # Unit tests
    └── test_cre.py         # CRE algorithm verification
```

## Technical Specifications

### DMS (Driver Monitoring System)

**Drowsiness Detection:**
- **EAR (Eye Aspect Ratio)**: 
  ```
  EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
  ```
  - Threshold: EAR < 0.25 for 3+ consecutive frames = drowsy
  
- **PERCLOS (Percentage Eye Closure)**:
  ```
  PERCLOS = (frames_eyes_closed / total_frames) × 100
  ```
  - 1-minute rolling window
  - Threshold: PERCLOS > 40% = severe drowsiness

**Head Pose Estimation:**
- Pitch (nodding): > 20° deviation = distraction
- Yaw (looking away): > 30° deviation = distraction
- Duration threshold: 2+ seconds

**Library**: MediaPipe Face Mesh (468 landmarks, 30+ FPS on CPU)

### ADAS (Advanced Driver Assistance)

**Lane Detection:**
- Canny edge detection → Hough transform
- ROI masking for road region
- Lane departure: Vehicle centroid > 15% offset from lane center

**Object Detection:**
- Model: YOLOv8-nano (3.2M params, ~30 FPS on CPU)
- Classes: car, truck, bus, motorcycle, bicycle, pedestrian
- Confidence threshold: 0.5

**Collision Warning:**
- Time-to-Collision (TTC):
  ```
  TTC = Distance / Relative_Velocity
  ```
- Warning levels: TTC < 3s (caution), TTC < 1.5s (critical)

### Context Risk Engine (CRE)

**Base Weights:**
| Parameter | Weight | Description |
|-----------|--------|-------------|
| Drowsiness | 0.30 | EAR + PERCLOS combined |
| Attention | 0.20 | Head pose deviation |
| Speed | 0.15 | Current velocity |
| Lane Position | 0.10 | Deviation from center |
| Forward Collision | 0.15 | TTC-based risk |
| Visibility | 0.10 | Weather/lighting factor |

**Context Multipliers:**

*Time of Day:*
| Time Range | Multiplier | Rationale |
|------------|------------|-----------|
| 00:00-05:00 | 1.5 | Peak drowsiness hours |
| 05:00-07:00 | 1.2 | Early morning |
| 07:00-22:00 | 1.0 | Normal hours |
| 22:00-00:00 | 1.3 | Late night |

*Zone Type:*
| Zone | Multiplier | Rationale |
|------|------------|-----------|
| Highway | 1.3 | High speed environment |
| Blackspot | 1.5 | Known accident location |
| School Zone | 1.4 | Pedestrian risk |
| Urban | 1.0 | Baseline |
| Rural | 1.2 | Limited infrastructure |

*Visibility:*
| Condition | Multiplier | Detection Method |
|-----------|------------|------------------|
| Clear | 1.0 | Baseline |
| Overcast | 1.1 | Brightness analysis |
| Rain | 1.3 | Edge clarity + brightness |
| Fog | 1.4 | Contrast ratio |
| Night | 1.3 | Ambient light level |

**Alert Escalation:**
| Risk Score | Level | Response |
|------------|-------|----------|
| 0.0-0.3 | Green | Normal - no alert |
| 0.3-0.5 | Yellow | Gentle audio reminder |
| 0.5-0.7 | Orange | Repeated beeps + visual |
| 0.7-0.85 | Red | Loud alarm + voice warning |
| 0.85-1.0 | Critical | Continuous alarm + brake prep |

## Demo Setup (March 28)

**Hardware for Demo:**
- Laptop with webcam (DMS feed)
- Same laptop screen split (YouTube road video for ADAS)
- Speakers for alerts

**Two-Feed Architecture:**
1. **Webcam → DMS Pipeline**: Your face → drowsiness + attention
2. **YouTube → ADAS Pipeline**: Dashcam video → lane + objects + collision

**Mock Data:**
- GPS: Simulated route through Pune with blackspot zones
- Time: Adjustable via UI for demo scenarios

**Demo Scenarios to Show:**
1. Normal driving (low risk score)
2. Drowsy driver in city (moderate alert)
3. Drowsy driver on highway at night near blackspot (maximum alert)
4. Lane departure detection
5. Forward collision warning

## Dependencies

```
# Core
opencv-python>=4.8.0
mediapipe>=0.10.0
ultralytics>=8.0.0  # YOLOv8
numpy>=1.24.0
PyYAML>=6.0

# Video feeds
yt-dlp>=2024.0.0
pafy>=0.5.5

# Interface
tkinter  # Usually pre-installed
Pillow>=10.0.0

# Audio
pygame>=2.5.0
pyttsx3>=2.90  # Text-to-speech for voice alerts

# Optional (for Pi deployment later)
# picamera2>=0.3.0
# RPi.GPIO>=0.7.0
```

## Coding Standards

1. **Type hints everywhere** - All functions must have type annotations
2. **Docstrings** - Google style for all public functions
3. **Error handling** - Graceful degradation, never crash during demo
4. **Logging** - Use Python logging module, not print statements
5. **Config-driven** - All magic numbers in config.yaml, not hardcoded
6. **Frame rate tracking** - Every module must report its FPS

## File Naming Convention

- Snake_case for files: `drowsiness_detector.py`
- PascalCase for classes: `DrowsinessDetector`
- snake_case for functions: `calculate_ear()`
- SCREAMING_SNAKE for constants: `EAR_THRESHOLD`

## Git Workflow

```bash
# Branch naming
feature/dms-drowsiness
feature/adas-lane-detection
feature/cre-risk-calculator
fix/ear-threshold-adjustment

# Commit message format
[MODULE] Brief description

# Examples:
[DMS] Add PERCLOS calculation with rolling window
[ADAS] Implement YOLOv8-nano object detection
[CRE] Add time-of-day context multiplier
[FIX] Adjust EAR threshold for better accuracy
```

## Success Criteria for Demo

1. ✅ Both feeds running simultaneously at 15+ FPS
2. ✅ Drowsiness detection triggers within 3 seconds of eye closure
3. ✅ Risk score updates in real-time on dashboard
4. ✅ Alert escalation visibly responds to context changes
5. ✅ Lane departure warning works on YouTube video
6. ✅ Object detection boxes visible on ADAS feed
7. ✅ Smooth transitions between alert levels
8. ✅ No crashes during 10-minute demo

## Known Limitations (Acknowledge in Presentation)

1. YouTube feed latency (~2-3 second delay)
2. Mock GPS instead of real GPS module
3. Weather detection is image-based approximation
4. Blackspot data is sample dataset, not complete India coverage
5. No actual vehicle integration (standalone demo)

## Post-Demo Roadmap (For Capstone Extension)

1. Raspberry Pi 5 deployment
2. Real GPS module integration
3. IR camera for night DMS
4. OBD-II integration for actual speed
5. Cloud dashboard for fleet management
6. Edge optimization with TensorRT

---

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download YOLOv8-nano model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Run the full system
python main.py

# Run individual modules for testing
python -m dms.drowsiness --test
python -m adas.object_detection --test
python -m context_engine.risk_calculator --test

# Run with specific YouTube video
python main.py --youtube-url "VIDEO_URL_HERE"
```

## Contact

### **Team Lead**: Vedant Shinde
### **Guide**: Asst. Prof. Amit Nehte 
### **Contributors**: 
- Vipul Deshmukh
- Shaunak Pandit
- Shubh Rajput 
### **Institution**: MIT-WPU, Pune
