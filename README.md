# Motion Detection and Face Tracking with Optical Flow and MediaPipe Face Mesh
This project is a Python-based application that detects motion using optical flow and tracks faces using MediaPipe's Face Mesh. The application captures motion in video frames, detects any faces, and then saves the motion, the detected face, and a merged image. It is designed to work with video input and provides real-time feedback on detected motion and faces.

# Features
Motion Detection: Uses Farneback optical flow to detect motion in the video stream.
Face Detection and Tracking: Utilizes MediaPipe's Face Mesh for detecting and tracking facial landmarks.
Frame Capture: Captures frames when motion is detected and saves images of the detected motion, the face, and a merged view of both.
Real-time Processing: Provides real-time feedback on motion and face detection with visualizations.

# Requirements
Python 3.x
OpenCV
MediaPipe
NumPy
Install the required packages using pip:

bash
pip install opencv-python mediapipe numpy
File Structure
The following folders are created to store the captured images:

Captured Motion: Stores images with detected motion.
If Any Faces: Stores images with detected faces.
Merged: Stores merged images of motion and face detection.
These folders are automatically created during runtime if they don't already exist.

# How It Works
Video Capture: The program reads a video file (Undetected_Video.mp4) using OpenCV.
Optical Flow Calculation: The Farneback method calculates the optical flow between consecutive frames to detect motion.
Motion Detection: If the motion magnitude exceeds a defined threshold (MOTION_THRESHOLD), it flags the frame as having detected motion.
Face Mesh Detection: If motion is detected, the frame is processed using MediaPipe's Face Mesh to detect facial landmarks. A bounding box is drawn around the detected face.
Frame Saving:
If motion and a face are detected, the following are saved:
The frame with detected motion.
The frame with the detected face.
A merged image containing both the grayscale motion image and the detected face.
The images are timestamped and stored in their respective folders.
Visualization: The application displays:
The current frame with motion and face annotations.
A visual representation of the optical flow (in HSV and vector form).
# Usage
Clone the repository:

bash
git clone <repository-url>
cd <repository-directory>
Run the script:

bash
python script_name.py
Press ESC to stop the video processing.

# Example Outputs
Motion Detected: A frame with the motion detected will be stored in the Captured Motion folder.
Face Detected: If a face is detected, the image will be saved in the If Any Faces folder.
Merged View: The merged grayscale motion image and detected face will be stored in the Merged folder.
Configuration
MOTION_THRESHOLD: This value determines the sensitivity of motion detection. Adjust the MOTION_THRESHOLD in the code to increase or decrease sensitivity.
python
MOTION_THRESHOLD = 5.0

# References
OpenCV Optical Flow
MediaPipe Face Mesh

# License
This project is licensed under the MIT License.
