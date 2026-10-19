"""
Biomedical Waste Detection - Production Inference Entry Point
Project: Computer Vision-Based Biomedical Waste Detection and Intelligent Segregation

Usage Examples:
    # Default test directory:
    python predict.py

    # Single image inference:
    python predict.py --source dataset/test/images/gauze_022_jpg.rf.432c8f61cb64ff11f654c2f95afe5e73.jpg

    # Custom confidence and output folder:
    python predict.py --source dataset/test/images --conf 0.35 --project runs/inference --name demo_test
"""

from src.predict import main

if __name__ == "__main__":
    main()
