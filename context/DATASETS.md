# CAVSS - Datasets, APIs, and Resources

## Pre-trained Models

### Face Detection & Landmarks

| Model | Source | Size | FPS (CPU) | Use Case |
|-------|--------|------|-----------|----------|
| MediaPipe Face Mesh | Google | ~2MB | 30+ | Primary - 468 landmarks |
| MediaPipe Face Detection | Google | ~1MB | 30+ | Backup face detector |

**Installation**: Comes with `mediapipe` package
```python
import mediapipe as mp
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

### Object Detection

| Model | Source | Size | FPS (CPU) | mAP | Use Case |
|-------|--------|------|-----------|-----|----------|
| YOLOv8n | Ultralytics | 6.3MB | 25-30 | 37.3 | Primary detector |
| YOLOv8s | Ultralytics | 22.5MB | 15-20 | 44.9 | Higher accuracy option |

**Download**:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Auto-downloads on first use
```

**Direct Download**: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

### Lane Detection

No pre-trained model needed - using classical CV:
- Canny edge detection
- Hough Line Transform
- Polynomial curve fitting

## Datasets

### Drowsiness Detection Training Data (Reference Only)

| Dataset | Size | Description | Link |
|---------|------|-------------|------|
| UTA-RLDD | 60 videos | Real-life drowsiness dataset | https://sites.google.com/view/utaborun/home |
| NTHU-DDD | 36 subjects | Driver drowsiness database | http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/ |
| YawDD | 322 videos | Yawning detection dataset | https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset |

**Note**: For demo, we use real-time webcam. These datasets are for future model fine-tuning.

### Road/Driving Datasets (Reference Only)

| Dataset | Description | Link |
|---------|-------------|------|
| BDD100K | 100K driving videos, diverse conditions | https://bdd-data.berkeley.edu/ |
| KITTI | German roads, well-annotated | http://www.cvlibs.net/datasets/kitti/ |
| Indian Driving Dataset | IIT-specific, Indian roads | https://idd.insaan.iiit.ac.in/ |

### Indian Blackspot Data

**Source**: Ministry of Road Transport and Highways (MoRTH)
**Format**: JSON with lat/long coordinates

```json
{
  "blackspots": [
    {
      "id": "MH-PUN-001",
      "name": "Katraj Ghat",
      "latitude": 18.4529,
      "longitude": 73.8553,
      "type": "ghat_section",
      "accident_count_2023": 47,
      "fatalities_2023": 12
    },
    {
      "id": "MH-PUN-002",
      "name": "Pune-Mumbai Expressway KM 42",
      "latitude": 18.7123,
      "longitude": 73.3456,
      "type": "highway",
      "accident_count_2023": 31,
      "fatalities_2023": 8
    }
  ]
}
```

**Sample Pune Blackspots** (include in `data/blackspots.json`):
- Katraj Ghat
- Navale Bridge
- Chandni Chowk
- Warje Malwadi
- Kharadi Bypass

## YouTube Video Sources for Demo

### Indian Highway Dashcam Videos

| Description | Duration | Quality | Notes |
|-------------|----------|---------|-------|
| Pune-Mumbai Expressway | 15+ min | 1080p | Good lane markings |
| NH48 Day Drive | 20+ min | 1080p | Mixed traffic |
| Bangalore-Mysore Highway | 30+ min | 1080p | Clear conditions |
| Mumbai Night Drive | 10+ min | 720p | Night visibility test |
| Ghat Section Drive | 15+ min | 1080p | Curves + warnings |

**Search Terms for Good Videos**:
- "Indian highway dashcam 4K"
- "Pune Mumbai expressway drive"
- "NH48 road trip dashcam"
- "Indian night driving dashcam"

**Video Requirements for Demo**:
- Resolution: 720p minimum
- Frame rate: 30 FPS preferred
- Content: Visible lane markings, other vehicles
- Duration: 10+ minutes for continuous demo

## APIs

### Weather/Visibility (Optional Enhancement)

| API | Free Tier | Use Case |
|-----|-----------|----------|
| OpenWeatherMap | 1000 calls/day | Current visibility data |
| WeatherAPI | 1M calls/month | Detailed weather |

**For Demo**: Use image-based visibility estimation instead of API:
```python
def estimate_visibility(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate contrast (standard deviation)
    contrast = gray.std()
    
    # Calculate brightness (mean)
    brightness = gray.mean()
    
    # Simple visibility score
    if contrast < 30 and brightness < 100:
        return "fog", 1.4
    elif brightness < 50:
        return "night", 1.3
    elif contrast < 40:
        return "overcast", 1.1
    else:
        return "clear", 1.0
```

### GPS/Location (Mock for Demo)

**For Demo**: Pre-recorded route with timestamps
```python
DEMO_ROUTE = [
    {"time": 0, "lat": 18.5204, "lon": 73.8567, "zone": "urban", "name": "Pune City"},
    {"time": 60, "lat": 18.4800, "lon": 73.8200, "zone": "highway", "name": "NH48 Entry"},
    {"time": 120, "lat": 18.4529, "lon": 73.8553, "zone": "blackspot", "name": "Katraj Ghat"},
    {"time": 180, "lat": 18.4200, "lon": 73.7800, "zone": "rural", "name": "Outskirts"},
    {"time": 240, "lat": 18.3900, "lon": 73.7500, "zone": "highway", "name": "Expressway"},
]
```

## Audio Resources

### Alert Sounds

| Sound | Purpose | Format |
|-------|---------|--------|
| gentle_chime.wav | Low risk alert | 440Hz, 0.3s |
| warning_beep.wav | Medium risk | 880Hz, 0.5s |
| alarm.wav | High risk | Multi-tone, 1s |
| critical_siren.wav | Critical | Continuous |

**Generation** (no external files needed):
```python
import numpy as np
import pygame

def generate_beep(frequency, duration, volume=0.5):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(2 * np.pi * frequency * t) * volume
    wave = (wave * 32767).astype(np.int16)
    stereo_wave = np.column_stack((wave, wave))
    return pygame.sndarray.make_sound(stereo_wave)
```

### Text-to-Speech Alerts

Using `pyttsx3` (offline, no API needed):
```python
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed

VOICE_ALERTS = {
    "drowsy_warning": "Warning. Drowsiness detected. Please take a break.",
    "lane_departure": "Lane departure warning.",
    "collision_warning": "Collision warning. Brake now.",
    "critical": "Critical alert. Pull over immediately.",
}
```

## Configuration Defaults

### EAR Thresholds (Eye Aspect Ratio)

```yaml
drowsiness:
  ear_threshold: 0.25          # Below this = eyes closed
  ear_consec_frames: 3         # Consecutive frames to confirm
  perclos_window: 60           # Seconds for PERCLOS calculation
  perclos_threshold: 0.40      # 40% = severe drowsiness
```

### Head Pose Thresholds

```yaml
attention:
  pitch_threshold: 20          # Degrees - nodding
  yaw_threshold: 30            # Degrees - looking away
  duration_threshold: 2.0      # Seconds before alert
```

### Lane Detection

```yaml
lane:
  roi_top_percent: 0.6         # Top of ROI (60% down from top)
  roi_bottom_percent: 1.0      # Bottom of ROI
  canny_low: 50                # Canny edge low threshold
  canny_high: 150              # Canny edge high threshold
  hough_threshold: 50          # Hough line threshold
  departure_threshold: 0.15    # 15% offset = departure
```

### Object Detection

```yaml
object_detection:
  model: "yolov8n.pt"
  confidence: 0.5
  iou_threshold: 0.45
  classes: [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
```

### Risk Calculator

```yaml
risk_weights:
  drowsiness: 0.30
  attention: 0.20
  speed: 0.15
  lane: 0.10
  collision: 0.15
  visibility: 0.10

context_multipliers:
  time:
    night_peak: 1.5      # 00:00-05:00
    night_late: 1.3      # 22:00-00:00
    morning_early: 1.2   # 05:00-07:00
    normal: 1.0          # 07:00-22:00
  
  zone:
    blackspot: 1.5
    school: 1.4
    highway: 1.3
    rural: 1.2
    urban: 1.0
  
  visibility:
    fog: 1.4
    rain: 1.3
    night: 1.3
    overcast: 1.1
    clear: 1.0

alert_thresholds:
  green: 0.3
  yellow: 0.5
  orange: 0.7
  red: 0.85
  critical: 1.0
```

## Hardware Requirements (Demo)

### Minimum (Laptop Demo)
- CPU: Intel i5 / Ryzen 5 or better
- RAM: 8GB
- Webcam: 720p, 30 FPS
- Storage: 2GB free
- OS: Windows 10/11, Ubuntu 20.04+, macOS 12+

### For Pi Deployment (Post-Demo)
- Raspberry Pi 5 (8GB recommended)
- Pi Camera Module 3
- USB IR Camera (for night DMS)
- GPS Module (NEO-6M)
- 3.5" LCD Display
- 5V 5A Power Supply

## Directory Structure After Setup

```
CAVSS/
├── models/
│   └── yolov8n.pt           # Downloaded automatically
├── data/
│   ├── blackspots.json      # Pune blackspot coordinates
│   └── demo_route.json      # Mock GPS route
├── sounds/                   # Generated at runtime
├── logs/                     # Runtime logs
└── output/                   # Recorded demo videos
```

## Troubleshooting

### Common Issues

1. **MediaPipe not detecting face**
   - Check lighting
   - Ensure webcam is not obstructed
   - Try `min_detection_confidence=0.3`

2. **YOLOv8 slow on CPU**
   - Use `yolov8n.pt` (nano version)
   - Reduce input resolution to 480p
   - Set `device='cpu'` explicitly

3. **YouTube feed not working**
   - Update yt-dlp: `pip install -U yt-dlp`
   - Check video availability
   - Try different video URL

4. **Audio alerts not playing**
   - Install pygame: `pip install pygame`
   - Check system audio
   - Initialize pygame.mixer before use

5. **High CPU usage**
   - Limit frame rate to 15 FPS
   - Process every 2nd frame
   - Reduce resolution

---

## Quick Reference Links

- MediaPipe Docs: https://developers.google.com/mediapipe
- YOLOv8 Docs: https://docs.ultralytics.com/
- OpenCV Docs: https://docs.opencv.org/
- yt-dlp Docs: https://github.com/yt-dlp/yt-dlp
- Pygame Docs: https://www.pygame.org/docs/
