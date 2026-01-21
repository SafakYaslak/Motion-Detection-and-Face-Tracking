# Motion Detection and Face Tracking

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)

[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange.svg)](https://google.github.io/mediapipe/)

An advanced Python application that performs motion detection and face tracking in video streams. Real-time image processing based on Optical Flow and MediaPipe.

## About the Project

This system detects motion in video images and automatically initiates face recognition when motion is detected. It records detected motions and faces as timestamped images.

### What is it used for?

- Security camera applications
- Automated surveillance systems
- Human activity tracking
- Smart home security systems
- Autonomous recording systems

## Features

### Motion Detection
- **Optical Flow Algorithm**: Pixel-level motion analysis using the Farneback method
- **Threshold Control**: Adjustable sensitivity (MOTION_THRESHOLD)
- **Real-Time Visualization**: HSV color map and vector flow representation
- **FPS Tracking**: Instant frame rate calculation

### Face Recognition
- **MediaPipe Face Mesh**: 468-point face landmark detection
- **Automatic Framing**: Automatically delimits the face area
- **High Accuracy**: Operates with a 30% confidence threshold

### Recording System
- **Three Folder Structure**: Organized file management
- **Time-Stamped Recordings**: Each image contains date and time information
- **Automatic Saving**: Every 5 frames Automatic recording
- **Combined Images**: Side-by-side recording of motion and face images

## Installation

### Requirements

```bash
Python 3.7+
Webcam or video file
```
### Steps

1. Clone the repository:
```bash
git clone https://github.com/SafakYaslak/Motion-Detection-and-Face-Tracking.git
cd Motion-Detection-and-Face-Tracking
```
2. Install the necessary libraries:
```bash
pip install -r requirements.txt
```
**requirements.txt:**
```
opencv-python>=4.5.0
mediapipe>=0.10.0
numpy>=1.19.0
```
3. Create the folder structure:
```bash
mkdir "Captured Motion"
mkdir "If Any Faces"
mkdir "Merged"
```

## Usage
### Basic Usage
```python
python motion_face_tracking.py
```
### Parameters
Adjustable parameters within the code:

```python
# Video source
cap = cv2.VideoCapture("Undetected_Video.mp4") # or 0 (for webcam)

# Motion sensitivity (lower value = more sensitive)
MOTION_THRESHOLD = 5.0

# Recording frequency (how many frames to record in)
if frame_counter % 5 == 0: # 5 frames = ~0.2 seconds

# MediaPipe parameters
max_num_faces=1 # Maximum number of faces
min_detection_confidence=0.3 # Detection confidence threshold
min_tracking_confidence=0.3 # Tracking confidence threshold
```
### Controls

- **ESC key**: Exit the program
- 3 windows open while the program is running:
- MediaPipe Face Mesh (Main image)
- Motion Track (HSV motion map)
- Flow (Vector flow representation)

## File Structure

```
Motion-Detection-and-Face-Tracking/
├── motion_face_tracking.py # Main program
├── Undetected_Video.mp4 # Test video file
├── Captured Motion/ # Images with detected motion
│ └── Detect_Motion*.jpg
├── If Any Faces/ # Detected faces
│ └── Detect_Face*.jpg
├── Merged/ # Merged images
│ └── Merged_Motion_and_Face*.jpg
├── requirements.txt # Python dependencies
└── README.md # This file
```
## Working Principle
### 1. Optical Flow Analysis

```python
flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
0.5, 3, 15, 3, 5, 1.2, 0)
```

**Farneback Parameters:**
- `pyr_scale=0.5`: Pyramid scale factor
- `levels=3`: Number of pyramid layers
- `winsize=15`: Window size
- `iterations=3`: Iterations per pyramid level
- `poly_n=5`: Pixel adjacency
- `poly_sigma=1.2`: Gaussian standard deviation

### 2. Motion Detection

```python
motion_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
if np.max(motion_magnitude) > MOTION_THRESHOLD:
motion_detected = True
```

**How ​​it Works:**

- Magnitude is calculated from the velocity vectors in the X and Y directions
- The maximum magnitude is compared with the threshold value
- Motion exceeding the threshold is detected

### 3. Face Recognition and Framing

```python
# Detects 468 landmarks with MediaPipe
results = face_mesh.process(image)

# Calculates rectangular coordinates from landmarks
min_x, max_x, min_y, max_y = face_rectangle_coordinates(image, landmarks)

# Crop face area
detect_face = captured_motion[min_y:max_y, min_x:max_x]
```
### 4. Recording System

**Three different outputs are produced:**

1. **Captured Motion**: Entire image + date/time stamp
2. **If Any Faces**: Only the face area (grayscale)
3. **Merged**: Side-by-side merged image

## Visualization Functions

### HSV Motion Map

```python
def draw_hsv(flow):
# Color = Motion direction
# Brightness = Motion speed
ang = np.arctan2(fy, fx) + np.pi
v = np.sqrt(fx*fx + fy*fy)
hsv[...,0] = ang * (180/np.pi/2) # Hue
hsv[...,2] = np.minimum(v*4, 255) # Value
```

**Color Encoding:**

- Red: Move right
- Blue: Move left
- Green: Move up
- Yellow: Move down

### Vector Flow Drawing

```python
def draw_flow(img, flow, step=16):
# Draws an arrow every 16 pixels
# Arrow length = Movement speed
# Arrow direction = Movement direction
```

## Performance

- **FPS**: Displayed in real time in the terminal
- **Example Performance**:
- 720p video: ~20-30 FPS
- 1080p video: ~15-25 FPS
- Webcam: ~25-35 FPS

## Output Examples

### Captured Motion
- Full-size image
- Date in top corner (YYYY-MM-DD)
- Time below (HH:MM:SS)
- "Motion Detected" text

### If Any Faces
- Face area only
- Grayscale format
- Framed with face landmarks

### Merged
- Left: Full image (grayscale)
- Right: Face area (resized)

## Usage Recommendations

### Adjusting Motion Sensitivity

```python
MOTION_THRESHOLD = 5.0 # Sensitive (small movements)
MOTION_THRESHOLD = 10.0 # Medium (normal movements)
MOTION_THRESHOLD = 20.0 # Less sensitive (large movements)
```

### Recording Frequency Settings

```python
frame_counter % 5 == 0 # Frequent recording (~5 FPS)
frame_counter % 10 == 0 # Moderate (~2.5 FPS)
frame_counter % 30 == 0 # Infrequent (~1 FPS)
```
### Webcam Usage
```python
cap = cv2.VideoCapture(0) # Default webcam
cap = cv2.VideoCapture(1) # Second camera
```
## Troubleshooting

### Folder Error
```
Error: Captured Motion folder not found
```
**Solution**: Create the necessary folders (see installation steps above)

### Low FPS
**Solution**:

- Reduce video resolution
- Reduce `max_num_faces` value
- Optimize optical flow parameters

### Face Detection Not working
**Solution**:
- Lower the `min_detection_confidence` value (e.g., 0.2)
- Improve lighting
- Adjust camera position

## Advanced Customization

### Multiple Face Recognition

```python
max_num_faces=5 # Simultaneous recognition of 5 faces
```
### Landmark Visualization

```python
mp_drawing.draw_landmarks(
image=image,
landmark_list=face_landmarks,
connections=mp_face_mesh.FACEMESH_TESSELATION,
landmark_drawing_spec=None,
connection_drawing_spec=mp_drawing_styles
.get_default_face_mesh_tesselation_style())
```

## Technical Details

### What is Optical Flow?
An algorithm that calculates pixel movements between two consecutive frames. Generates a vector (dx, dy) for each pixel.

### MediaPipe Face Mesh
Machine learning model developed by Google. Detects 468 3D face landmarks in real time.

### HSV Color Space
- **H** (Hue): Direction of movement (0-180°)
- **S** (Saturation): Constant (255)
- **V** (Value): Magnitude of movement (0-255)

## Contributing

1. Fork
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit (`git commit -m 'Add new feature'`)
4. Push (`git push origin feature/new-feature`)
5. Open a Pull Request


## Contact

**Safak Yaslak**

- GitHub: [@SafakYaslak](https://github.com/SafakYaslak)

## Thanks

If you found the project helpful, don't forget to give it a star rating!

---

**Note**: Before the first run, make sure the "Captured Motion", "If Any Faces" and "Merged" folders are created. Update your video file name in the code or use `VideoCapture(0)` for the webcam.
