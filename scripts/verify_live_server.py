"""
Verify live web application health, endpoints, and inference without restarting server.
"""

import urllib.request
import json
import base64
from pathlib import Path

BASE_URL = "http://127.0.0.1:7860"

def test_live_app():
    print("=" * 60)
    print("TESTING LIVE WEB APPLICATION HEALTH & ENDPOINTS")
    print("=" * 60)
    
    # 1. Info endpoint
    print("Testing GET /api/info...")
    req_info = urllib.request.Request(f"{BASE_URL}/api/info")
    with urllib.request.urlopen(req_info) as res:
        info_data = json.loads(res.read().decode())
        print(f"  HTTP Status : {res.status}")
        print(f"  Model Loaded: {info_data.get('model_name')}")
        print(f"  Weights Path: {info_data.get('weights_path')}")
        print(f"  Class Count : {info_data.get('num_classes')}")
        print(f"  Device      : {info_data.get('device')}")
        
    # 2. Image upload / prediction endpoint
    print("\nTesting POST /api/predict (Image Upload simulation)...")
    img_path = Path("dataset/sample_test_images/gloves_009_jpg.rf.83286b4512232845714a3d27e98728a7.jpg")
    b64_str = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    
    payload = {
        "image": f"data:image/jpeg;base64,{b64_str}",
        "conf": 0.25,
        "iou": 0.45,
        "imgsz": 640
    }
    req_predict = urllib.request.Request(
        f"{BASE_URL}/api/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_predict) as res:
        pred_data = json.loads(res.read().decode())
        print(f"  HTTP Status     : {res.status}")
        print(f"  Inference Time  : {pred_data.get('inference_ms')} ms")
        print(f"  Detections Count: {pred_data.get('num_detections')}")
        for d in pred_data.get("detections", []):
            print(f"    - {d['class_name']} ({d['confidence']*100:.1f}%) -> {d['segregation']['bin']}")
            
    # 3. Webcam streaming frame simulation (calls /api/predict with imgsz: 416)
    print("\nTesting POST /api/predict with Webcam Frame Payload (imgsz=416)...")
    payload_webcam = {
        "image": f"data:image/jpeg;base64,{b64_str}",
        "conf": 0.25,
        "iou": 0.45,
        "imgsz": 416
    }
    req_stream = urllib.request.Request(
        f"{BASE_URL}/api/predict",
        data=json.dumps(payload_webcam).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_stream) as res:
        stream_data = json.loads(res.read().decode())
        print(f"  HTTP Status     : {res.status}")
        print(f"  Inference Time  : {stream_data.get('inference_ms')} ms")
        print(f"  Detections Count: {stream_data.get('num_detections')}")
        print(f"  Has Visual Frame: {bool(stream_data.get('annotated_image'))}")
        
    print("\nALL SERVER ENDPOINTS FULLY VERIFIED HEALTHY & RESPONSIVE!")

if __name__ == "__main__":
    test_live_app()
