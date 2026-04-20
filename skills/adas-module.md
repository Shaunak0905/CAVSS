---
name: adas-module
description: |
  Build the Advanced Driver Assistance System module for CAVSS. Use this skill when working on:
  - Object detection with YOLOv8
  - Lane detection and departure warning
  - Collision warning (TTC calculation)
  - Road scene analysis
  Trigger on: "ADAS", "object detection", "YOLO", "lane detection", "collision warning", "TTC", "vehicle detection", "pedestrian detection", "lane departure"
---

# ADAS Module Development Skill

## Overview

The ADAS module processes the road-facing camera (YouTube feed for demo) to detect:
1. Objects (vehicles, pedestrians, cyclists)
2. Lane markings
3. Potential collision risks

## Object Detection with YOLOv8

### Setup

```python
from ultralytics import YOLO
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # COCO classes we care about for driving
        self.target_classes = {
            0: "person",
            1: "bicycle", 
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in self.target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=self.target_classes[cls_id],
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=((x1+x2)//2, (y1+y2)//2),
                    area=(x2-x1) * (y2-y1)
                ))
        
        return detections
```

### Detection Data Structure

```python
@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int
    distance_estimate: Optional[float] = None  # meters
    ttc: Optional[float] = None  # time to collision
```

## Lane Detection

### Classical CV Approach (No ML needed)

```python
class LaneDetector:
    def __init__(self, frame_shape):
        self.h, self.w = frame_shape[:2]
        self.roi_vertices = self._define_roi()
        
        # Lane history for smoothing
        self.left_lane_history = deque(maxlen=5)
        self.right_lane_history = deque(maxlen=5)
    
    def _define_roi(self) -> np.ndarray:
        """Define region of interest (trapezoid on road)."""
        return np.array([[
            (int(self.w * 0.1), self.h),           # Bottom left
            (int(self.w * 0.4), int(self.h * 0.6)), # Top left
            (int(self.w * 0.6), int(self.h * 0.6)), # Top right
            (int(self.w * 0.9), self.h)            # Bottom right
        ]], dtype=np.int32)
    
    def detect(self, frame: np.ndarray) -> LaneResult:
        # 1. Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Canny edge detection
        edges = cv2.Canny(blur, 50, 150)
        
        # 4. Apply ROI mask
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, self.roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # 5. Hough Line Transform
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=2,
            theta=np.pi/180,
            threshold=50,
            minLineLength=40,
            maxLineGap=100
        )
        
        # 6. Separate left and right lanes
        left_lines, right_lines = self._separate_lanes(lines)
        
        # 7. Average and extrapolate
        left_lane = self._average_lane(left_lines, "left")
        right_lane = self._average_lane(right_lines, "right")
        
        # 8. Calculate lane center and vehicle offset
        lane_center, vehicle_offset = self._calculate_offset(left_lane, right_lane)
        
        return LaneResult(
            left_lane=left_lane,
            right_lane=right_lane,
            lane_center=lane_center,
            vehicle_offset=vehicle_offset,
            departure_warning=abs(vehicle_offset) > 0.15
        )
    
    def _separate_lanes(self, lines) -> Tuple[List, List]:
        """Separate lines into left and right based on slope."""
        left_lines = []
        right_lines = []
        
        if lines is None:
            return left_lines, right_lines
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter out near-horizontal lines
            if abs(slope) < 0.3:
                continue
            
            if slope < 0:  # Negative slope = left lane
                left_lines.append(line[0])
            else:  # Positive slope = right lane
                right_lines.append(line[0])
        
        return left_lines, right_lines
    
    def _average_lane(self, lines, side: str) -> Optional[Tuple]:
        """Average multiple line segments into single lane line."""
        if not lines:
            return None
        
        x_coords = []
        y_coords = []
        
        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        # Fit polynomial
        poly = np.polyfit(y_coords, x_coords, 1)
        
        # Extrapolate to ROI boundaries
        y1 = self.h
        y2 = int(self.h * 0.6)
        x1 = int(np.polyval(poly, y1))
        x2 = int(np.polyval(poly, y2))
        
        # Add to history for smoothing
        history = self.left_lane_history if side == "left" else self.right_lane_history
        history.append((x1, y1, x2, y2))
        
        # Return averaged line
        if len(history) > 0:
            avg_line = np.mean(history, axis=0).astype(int)
            return tuple(avg_line)
        
        return (x1, y1, x2, y2)
    
    def _calculate_offset(self, left_lane, right_lane) -> Tuple[int, float]:
        """Calculate lane center and vehicle offset."""
        frame_center = self.w // 2
        
        if left_lane and right_lane:
            # Both lanes detected
            left_x = left_lane[0]  # x at bottom
            right_x = right_lane[0]
            lane_center = (left_x + right_x) // 2
        elif left_lane:
            # Only left lane - estimate center
            lane_center = left_lane[0] + 300  # Approximate lane width
        elif right_lane:
            # Only right lane - estimate center
            lane_center = right_lane[0] - 300
        else:
            # No lanes detected
            return frame_center, 0.0
        
        # Offset as fraction of lane width
        lane_width = 600  # Approximate pixels
        offset = (frame_center - lane_center) / lane_width
        
        return lane_center, offset
```

### Lane Result Structure

```python
@dataclass
class LaneResult:
    left_lane: Optional[Tuple[int, int, int, int]]
    right_lane: Optional[Tuple[int, int, int, int]]
    lane_center: int
    vehicle_offset: float  # -1 to 1, negative = left of center
    departure_warning: bool
```

## Collision Warning

### Time-to-Collision (TTC) Estimation

```python
class CollisionWarning:
    def __init__(self):
        self.previous_detections = {}
        self.frame_time = 1/15  # Assuming 15 FPS
        
        # Warning thresholds
        self.TTC_WARNING = 3.0  # seconds
        self.TTC_CRITICAL = 1.5  # seconds
    
    def estimate_distance(self, bbox: Tuple, frame_height: int) -> float:
        """
        Estimate distance based on bounding box size.
        Assumes average car height of 1.5m.
        """
        _, y1, _, y2 = bbox
        box_height = y2 - y1
        
        # Reference: car at 10m has ~150px height (calibrate for your camera)
        REFERENCE_HEIGHT_PX = 150
        REFERENCE_DISTANCE_M = 10
        
        if box_height < 10:
            return float('inf')
        
        distance = (REFERENCE_HEIGHT_PX * REFERENCE_DISTANCE_M) / box_height
        return distance
    
    def calculate_ttc(self, detection: Detection, detection_id: str) -> Optional[float]:
        """
        Calculate Time-to-Collision for a detection.
        """
        current_distance = self.estimate_distance(detection.bbox, 720)
        
        if detection_id in self.previous_detections:
            prev_distance = self.previous_detections[detection_id]
            
            # Relative velocity (positive = approaching)
            relative_velocity = (prev_distance - current_distance) / self.frame_time
            
            if relative_velocity > 0.5:  # Approaching at >0.5 m/s
                ttc = current_distance / relative_velocity
                self.previous_detections[detection_id] = current_distance
                return ttc
        
        self.previous_detections[detection_id] = current_distance
        return None
    
    def assess_risk(self, detections: List[Detection]) -> CollisionRisk:
        """
        Assess overall collision risk from all detections.
        """
        min_ttc = float('inf')
        closest_threat = None
        
        for i, det in enumerate(detections):
            # Only check objects in center of frame (in our path)
            center_x = det.center[0]
            if 0.3 * 1280 < center_x < 0.7 * 1280:  # Center 40% of frame
                ttc = self.calculate_ttc(det, f"{det.class_name}_{i}")
                if ttc and ttc < min_ttc:
                    min_ttc = ttc
                    closest_threat = det
        
        return CollisionRisk(
            ttc=min_ttc if min_ttc < float('inf') else None,
            risk_level=self._get_risk_level(min_ttc),
            threat_object=closest_threat
        )
    
    def _get_risk_level(self, ttc: float) -> str:
        if ttc < self.TTC_CRITICAL:
            return "critical"
        elif ttc < self.TTC_WARNING:
            return "warning"
        else:
            return "safe"
```

## Module Structure

```
adas/
├── __init__.py
├── object_detection.py   # YOLOv8 wrapper
├── lane_detection.py     # Classical CV lane detection
├── collision_warning.py  # TTC calculation
└── adas_processor.py     # Main orchestrator
```

## Output Format

```python
@dataclass
class ADASOutput:
    """Output from ADAS module for CRE."""
    # Object detection
    detections: List[Detection]
    vehicle_count: int
    pedestrian_count: int
    
    # Lane detection
    lane_result: LaneResult
    lane_departure_risk: float  # 0-1
    
    # Collision warning
    collision_risk: CollisionRisk
    forward_collision_risk: float  # 0-1
    
    # Visibility estimate
    visibility_score: float  # 0-1 (1 = clear)
    visibility_condition: str  # "clear", "rain", "fog", etc.
    
    # Performance
    processing_time_ms: float
```

## Visibility Estimation

```python
def estimate_visibility(frame: np.ndarray) -> Tuple[str, float]:
    """
    Estimate visibility conditions from frame analysis.
    
    Returns:
        condition: str ("clear", "overcast", "fog", "rain", "night")
        multiplier: float (1.0 = clear, higher = worse)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Metrics
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Decision logic
    if brightness < 50:
        return "night", 1.3
    elif contrast < 25:
        return "fog", 1.4
    elif contrast < 35 and brightness < 120:
        return "overcast", 1.1
    elif contrast < 40:
        return "rain", 1.3
    else:
        return "clear", 1.0
```

## Performance Optimization

1. **Resize frame** before YOLO inference (640x480 is enough)
2. **Skip frames** - process every 2nd frame for lane detection
3. **ROI masking** - only process relevant areas
4. **Batch detections** - YOLO can process batches if needed
5. **Use threads** - lane detection and object detection can run in parallel

## Error Handling

```python
# If YouTube feed drops
if frame is None:
    return ADASOutput(
        detections=[],
        lane_result=LaneResult(None, None, frame_width//2, 0.0, False),
        collision_risk=CollisionRisk(None, "safe", None),
        # ... safe defaults
    )
```
