# Models

YOLOv8-nano downloads automatically on first run via ultralytics:

```python
from ultralytics import YOLO
YOLO('yolov8n.pt')  # auto-downloads ~6MB
```

Or manually:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```
