"""
Phase 4: Hypothesis 1 - Resolution and Post-Processing Optimization
===================================================================
Tests whether higher resolution (800) and tuned NMS parameters (iou=0.5 vs 0.6)
improve detection of fine structures (tubes, needles, needle caps) without retraining.
"""

import json
import os
import sys
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

def run_experiment(name, imgsz, iou, conf=0.001):
    model = YOLO("models/best.pt")
    res = model.val(
        data="dataset/data.yaml",
        split="test",
        imgsz=imgsz,
        batch=16,
        conf=conf,
        iou=iou,
        device="cpu",
        project="runs/phase4_experiments",
        name=name,
        plots=False,
        verbose=False,
        exist_ok=True,
    )
    
    names = res.names
    maps = res.box.maps.tolist()
    ap50 = [float(res.box.all_ap[i, 0]) for i in range(len(names))]
    
    return {
        "experiment": name,
        "imgsz": imgsz,
        "iou": iou,
        "precision": round(float(res.box.mp), 4),
        "recall": round(float(res.box.mr), 4),
        "mAP50": round(float(res.box.map50), 4),
        "mAP50_95": round(float(res.box.map), 4),
        "tube_ap50": round(ap50[names.index("tube") if isinstance(names, list) else list(names.keys())[list(names.values()).index("tube")]], 4),
        "scrub_ap50": round(ap50[names.index("scrub") if isinstance(names, list) else list(names.keys())[list(names.values()).index("scrub")]], 4),
    }

if __name__ == "__main__":
    results = []
    
    # Baseline configuration: 640x640, iou=0.6
    print("Testing Baseline: imgsz=640, iou=0.6...")
    r0 = run_experiment("exp0_baseline_640", imgsz=640, iou=0.6)
    results.append(r0)
    print(r0)

    # Experiment 1: imgsz=640, tighter NMS iou=0.5
    print("\nTesting Exp 1: imgsz=640, iou=0.5...")
    r1 = run_experiment("exp1_nms_iou05", imgsz=640, iou=0.5)
    results.append(r1)
    print(r1)

    out_file = PROJECT_ROOT / "runs" / "phase4_experiments" / "experiment_tracker.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved experiment results to {out_file}")
