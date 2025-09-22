# Stereo Visual Odometry

A compact implementation of a **stereo visual odometry (SVO)** pipeline using OpenCV and NumPy. Developed as part of my * CSE 568 Robotics Algorithms* coursework (SUNY Buffalo). The system estimates camera motion from KITTI stereo sequences by computing disparity/depth, extracting/tracking features, and solving frame-to-frame pose.

## Project Files
- `code/visual_odometry.py` — two-frame VO and full SVO with CSV export + trajectory plot
- `config.yaml` — paths and run settings
- `requirements.txt` — dependencies


##  What this does

- Loads KITTI stereo pairs (left/right) and **computes disparity** → **depth map**  
- Extracts **SIFT** features, performs **BFMatcher KNN** matches + **Lowe ratio** filtering  
- Forms **3D–2D correspondences** (3D from depth of frame _t_, 2D from frame _t+1_)  
- Estimates motion via **PnP + RANSAC**, integrates to build a **trajectory**  
- compares estimated trajectory with **KITTI ground truth** and plots





## Results 
-Visualization for the first 1000 frames















