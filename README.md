Real-Time Object Tracking System
A computer vision project that implements various object tracking algorithms for real-time object detection and tracking using OpenCV.

📌 Project Overview
This system provides multiple object tracking implementations using different OpenCV tracking algorithms. It allows users to select an object in the first frame and track it throughout the video sequence.

🎯 Features
Multi-Algorithm Support: Implements 8 different tracking algorithms

Real-time Tracking: Works with webcam, video files, and image sequences

Performance Comparison: Compare different tracking algorithms

Visual Feedback: Real-time visualization of tracking performance

FPS Monitoring: Track processing speed for each algorithm

📋 Supported Tracking Algorithms
BOOSTING Tracker - Based on online AdaBoost algorithm

MIL Tracker - Multiple Instance Learning tracker

KCF Tracker - Kernelized Correlation Filters

TLD Tracker - Tracking-Learning-Detection

MEDIANFLOW Tracker - Good for predictable motion

MOSSE Tracker - Minimum Output Sum of Squared Error

CSRT Tracker - Discriminative Correlation Filter with Channel and Spatial Reliability

GOTURN Tracker - Deep learning based tracker (requires model)

🚀 Quick Start
Installation
bash
# Clone the repository
git clone https://github.com/yourusername/object_tracking.git
cd object_tracking

# Install dependencies
pip install -r requirements.txt
