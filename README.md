# Hand Gesture Recognition using MediaPipe

This project implements a real-time hand gesture recognition system using MediaPipe and OpenCV. It detects hand landmarks in a video feed and recognizes predefined gestures based on the relative positions of the landmarks.

## Theory

Hand gesture recognition is a vital component in human-computer interaction, enabling intuitive and touch-free control systems. By analyzing the positions and movements of hand landmarks, specific gestures can be identified and translated into commands.

## MediaPipe Hands Framework

MediaPipe Hands is a high-fidelity hand and finger tracking solution. It identifies 21 3D landmarks for each detected hand and provides a robust foundation for gesture recognition. This framework uses machine learning models to process input images, detect hands, and return normalized landmark coordinates. Key advantages include:

- High accuracy and efficiency.
- Cross-platform support.
- Integration with various development environments.

## Gesture Recognition

Gestures are recognized by analyzing relative positions of hand landmarks. For example, the distance between the thumb and index finger can indicate a pinch or point gesture. Combining multiple landmarks allows for more complex gesture detection. The project employs the Euclidean distance formula to compute relationships between landmarks and define gesture rules.

## Applications

Hand gesture recognition has diverse applications, such as:

- Virtual reality and augmented reality interactions.
- Sign language interpretation.
- Touchless interfaces in healthcare and automotive industries.
- Gaming and entertainment.

## Features

- Real-time detection of hand landmarks using MediaPipe.
- Recognition of various gestures such as:
  - Thumbs Up
  - Peace
  - Pointing
  - Flat Hand
  - Thanks
  - Thumbs Down
  - Unknown gestures
- Visual feedback with landmarks and gesture names displayed on the video feed.

## Installation

### Prerequisites

- Python 3.7 or higher.
- Libraries:
  - OpenCV
  - MediaPipe

Install the required libraries using pip:

bash
```pip install opencv-python mediapipe```

##Usage
-Clone this repository and navigate to the project directory:

bash
```git clone <https://github.com/aruljothi156/Hand_Gesture_Recognition.git>```

##Run the Script:

bash
```python Handgesture.py```
Allow access to your webcam. The application will start capturing video and detecting gestures.

Press the 'q' key to exit the application.

---
##Code Explanation

###Key Components
-MediaPipe Hands Initialization:


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils 

*Configures the MediaPipe Hands model for detecting and tracking hands in the video feed.

## Gesture Recognition

Landmarks of the hand are extracted using MediaPipe.

Gestures are recognized based on the relative positions of key landmarks such as the thumb tip, index finger tip, etc.

A mapping dictionary (`gesture_map`) translates gesture IDs to descriptive text.

### Real-time Video Processing

The script captures video frames, processes them to detect hands, and overlays gesture recognition results on the frame.

### Gesture Mapping Logic

The function `recognize_gesture` uses relative distances and positions of landmarks to identify gestures.

Example:

- **Thumbs Up**: The thumb tip is above the other fingers and distant from the index finger.
- **Peace**: The thumb is below and distant from the index finger, with the index and middle fingers pointing upwards.

### Output

The application displays the webcam feed with hand landmarks and recognized gesture names overlaid on the video.




