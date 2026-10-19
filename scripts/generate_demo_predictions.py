"""
Generate comprehensive demo predictions for Biomedical Waste AI.
Verifies models/best.pt, evaluates ground truth vs predictions, and outputs visual artifacts in outputs/demo/.
"""

import os
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.predict import BIOMEDICAL_WASTE_CLASSES, resolve_weights

def main():
    model_path = resolve_weights("models/best.pt")
    print(f"[Model] Verified weights path: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"[Model] Class count: {len(model.names)}")
    print(f"[Model] Class map: {model.names}")
    
    output_dir = Path("outputs/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 14 classes name mapping
    class_names = model.names
    
    # Load label mapping from dataset
    test_img_dir = Path("dataset/test/images")
    test_lbl_dir = Path("dataset/test/labels")
    
    # Select 4 representative test images
    sample_files = [
        "syringe_015_jpg.rf.cd8eac031a8a91d3199ca951cf2c4954.jpg",
        "mask_002_jpg.rf.94129827eceec4062e5aaeacca35f1de.jpg",
        "gloves_009_jpg.rf.83286b4512232845714a3d27e98728a7.jpg",
        "needles_015_jpg.rf.8a8d25d22c03f08d2e17c0d2b96eb64d.jpg",
    ]
    
    results_summary = []
    
    for filename in sample_files:
        img_path = test_img_dir / filename
        if not img_path.exists():
            img_path = Path("dataset/sample_test_images") / filename
        if not img_path.exists():
            print(f"Skipping {filename}, not found")
            continue
            
        lbl_path = test_lbl_dir / (img_path.stem + ".txt")
        gt_classes = []
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cid = int(parts[0])
                        gt_classes.append(class_names.get(cid, str(cid)))
        
        # Run inference
        res = model.predict(source=str(img_path), conf=0.25, iou=0.45, imgsz=640, verbose=False)[0]
        
        # Save annotated image
        annotated_bgr = res.plot(conf=True, labels=True, font_size=0.8, line_width=2)
        save_path = output_dir / f"pred_{img_path.stem[:25]}.jpg"
        cv2.imwrite(str(save_path), annotated_bgr)
        
        # Copy input image for side-by-side
        input_copy_path = output_dir / f"orig_{img_path.stem[:25]}.jpg"
        orig_img = cv2.imread(str(img_path))
        cv2.imwrite(str(input_copy_path), orig_img)
        
        preds = []
        for box in res.boxes:
            cid = int(box.cls[0].item())
            cname = class_names.get(cid, str(cid))
            conf = float(box.conf[0].item())
            preds.append({"class": cname, "conf": round(conf, 4), "xyxy": [round(v, 1) for v in box.xyxy[0].tolist()]})
            
        pred_classes = [p["class"] for p in preds]
        
        # Evaluation comparison
        tp = [c for c in pred_classes if c in gt_classes]
        fp = [c for c in pred_classes if c not in gt_classes]
        fn = [c for c in gt_classes if c not in pred_classes]
        
        results_summary.append({
            "image": filename,
            "orig_saved": str(input_copy_path),
            "pred_saved": str(save_path),
            "ground_truth": gt_classes,
            "predictions": preds,
            "true_positives": tp,
            "false_positives": fp,
            "missed_detections": fn
        })
        
        print(f"\n--- Results for {filename[:30]} ---")
        print(f"Ground Truth : {gt_classes}")
        print(f"Predictions  : {[p['class'] + ' (' + str(round(p['conf']*100, 1)) + '%)' for p in preds]}")
        print(f"Correct (TP) : {tp}")
        print(f"False Pos(FP): {fp}")
        print(f"Missed (FN)  : {fn}")

    # Optional custom demo image check
    custom_demo_path = Path("dataset/sample_test_images/gloves_009_jpg.rf.83286b4512232845714a3d27e98728a7.jpg")
    if custom_demo_path.exists():
        print(f"\n--- Testing Standard Demo Image ({custom_demo_path.name}) ---")
        u_res = model.predict(source=str(custom_demo_path), conf=0.15, imgsz=640, verbose=False)[0]
        u_annotated = u_res.plot(conf=True, labels=True, font_size=0.8, line_width=2)
        u_save = output_dir / "demo_sample_pred.jpg"
        cv2.imwrite(str(u_save), u_annotated)
        
        u_preds = []
        for box in u_res.boxes:
            cid = int(box.cls[0].item())
            cname = class_names.get(cid, str(cid))
            conf = float(box.conf[0].item())
            u_preds.append({"class": cname, "conf": round(conf, 4), "xyxy": [round(v, 1) for v in box.xyxy[0].tolist()]})
        print(f"Sample detections at conf=0.15: {[p['class'] + ' (' + str(round(p['conf']*100, 1)) + '%)' for p in u_preds]}")

    summary_file = output_dir / "demo_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n[Complete] Summary and images saved to {output_dir}")

if __name__ == "__main__":
    main()
