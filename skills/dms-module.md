---
name: dms-module
description: |
  Build the Driver Monitoring System module for CAVSS. Use this skill when working on:
  - Drowsiness detection (EAR, PERCLOS)
  - Face mesh landmark extraction
  - Head pose estimation
  - Attention/distraction detection
  - Yawn detection
  Trigger on: "drowsiness", "DMS", "driver monitoring", "eye tracking", "face mesh", "head pose", "attention detection", "PERCLOS", "EAR"
---

# DMS Module Development Skill

## Overview

The DMS (Driver Monitoring System) uses MediaPipe Face Mesh to track 468 facial landmarks in real-time, then calculates drowsiness and attention metrics.

## Key Algorithms

### Eye Aspect Ratio (EAR)

```python
def calculate_ear(eye_landmarks: List[Tuple[float, float]]) -> float:
    """
    Calculate Eye Aspect Ratio from 6 eye landmarks.
    
    Landmarks order: [outer_corner, upper_1, upper_2, inner_corner, lower_1, lower_2]
    
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    
    Returns:
        float: EAR value (typically 0.15-0.35 for open eyes)
    """
    # Vertical distances
    v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    
    # Horizontal distance
    h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    
    ear = (v1 + v2) / (2.0 * h) if h > 0 else 0
    return ear
```

### PERCLOS Calculation

```python
class PERCLOSCalculator:
    """
    PERCLOS = Percentage of Eye Closure over time
    Industry standard: 40% closure in 1 minute = drowsy
    """
    
    def __init__(self, window_seconds: int = 60, fps: int = 30):
        self.window_size = window_seconds * fps
        self.eye_states = deque(maxlen=self.window_size)
        self.ear_threshold = 0.25
    
    def update(self, ear: float) -> float:
        """Add new EAR reading and return current PERCLOS."""
        is_closed = ear < self.ear_threshold
        self.eye_states.append(1 if is_closed else 0)
        
        if len(self.eye_states) < 30:  # Need minimum samples
            return 0.0
        
        return sum(self.eye_states) / len(self.eye_states)
```

### MediaPipe Landmark Indices

```python
# Eye landmarks for EAR calculation
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Mouth landmarks for yawn detection
MOUTH_OUTER = [61, 291, 0, 17]  # Left, right, top, bottom

# Head pose landmarks (for solvePnP)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Nose tip for head pose
NOSE_TIP = 1

# Reference points for head pose
HEAD_POSE_POINTS = [1, 33, 263, 61, 291, 199]  # Nose, eyes, mouth corners, chin
```

### Head Pose Estimation

```python
def estimate_head_pose(landmarks, frame_shape):
    """
    Estimate head pose using solvePnP.
    
    Returns:
        pitch, yaw, roll in degrees
    """
    # 3D model points (generic face model)
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (-30.0, -125.0, -30.0),   # Left eye corner
        (30.0, -125.0, -30.0),    # Right eye corner
        (-60.0, -70.0, -60.0),    # Left mouth corner
        (60.0, -70.0, -60.0),     # Right mouth corner
        (0.0, -330.0, -65.0)      # Chin
    ], dtype=np.float64)
    
    # Camera matrix (approximate)
    h, w = frame_shape[:2]
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Get 2D image points from landmarks
    image_points = get_landmark_coords(landmarks, HEAD_POSE_POINTS, frame_shape)
    
    # Solve PnP
    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, None
    )
    
    if not success:
        return 0, 0, 0
    
    # Convert rotation vector to Euler angles
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = np.hstack((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    
    pitch = euler_angles[0][0]
    yaw = euler_angles[1][0]
    roll = euler_angles[2][0]
    
    return pitch, yaw, roll
```

## Module Structure

```
dms/
├── __init__.py
├── drowsiness.py      # EAR + PERCLOS + yawn detection
├── attention.py       # Head pose + distraction tracking
├── face_mesh.py       # MediaPipe wrapper
└── dms_processor.py   # Main orchestrator
```

## Implementation Checklist

- [ ] Initialize MediaPipe Face Mesh with correct parameters
- [ ] Extract eye landmarks and calculate EAR
- [ ] Implement PERCLOS with rolling window
- [ ] Add yawn detection via Mouth Aspect Ratio
- [ ] Implement head pose estimation
- [ ] Track distraction duration
- [ ] Return normalized scores (0-1) for CRE
- [ ] Handle face not detected gracefully
- [ ] Add FPS tracking
- [ ] Implement smoothing to reduce jitter

## Output Format

```python
@dataclass
class DMSOutput:
    """Output from DMS module for CRE."""
    drowsiness_score: float  # 0-1, combined EAR+PERCLOS
    attention_score: float   # 0-1, head pose deviation
    ear_left: float
    ear_right: float
    perclos: float
    head_pitch: float
    head_yaw: float
    head_roll: float
    is_yawning: bool
    face_detected: bool
    processing_time_ms: float
```

## Error Handling

```python
# Always return safe defaults if face not detected
if not face_detected:
    return DMSOutput(
        drowsiness_score=0.0,  # Don't trigger false alarm
        attention_score=0.0,
        face_detected=False,
        # ... other defaults
    )
```

## Performance Tips

1. Use `refine_landmarks=True` for better eye tracking accuracy
2. Process at 15-20 FPS, not 30 - save CPU for other modules
3. Use numpy vectorized operations for distance calculations
4. Cache the camera matrix - it doesn't change
5. Apply exponential smoothing to reduce jitter in readings
