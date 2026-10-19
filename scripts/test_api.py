import urllib.request
import json
import base64
from pathlib import Path

img_path = Path("dataset/sample_test_images/gloves_009_jpg.rf.83286b4512232845714a3d27e98728a7.jpg")
b64_str = base64.b64encode(img_path.read_bytes()).decode("utf-8")

payload = {
    "image": f"data:image/jpeg;base64,{b64_str}",
    "conf": 0.25,
    "iou": 0.45,
    "imgsz": 640
}

req = urllib.request.Request(
    "http://127.0.0.1:7860/api/predict",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req) as res:
    data = json.loads(res.read().decode())
    print("HTTP Status:", res.status)
    print("Inference time:", data.get("inference_ms"), "ms")
    print("Detections count:", data.get("num_detections"))
    for d in data.get("detections", []):
        print(f" - {d['class_name']} ({d['confidence']*100:.1f}%) -> {d['segregation']['bin']} | {d['segregation']['disposal']}")
