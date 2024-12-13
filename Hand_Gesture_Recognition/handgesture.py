import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Gesture mapping dictionary
gesture_map = {
    0: "Thumbs Up",
    1: "Peace",
    2: "Pointing",
    3: "Pointing",
    4: "Flat Hand",
    5: "Thanks",
    6: "Thumbs Down"
}

# Calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Recognize gesture based on landmarks
def recognize_gesture(hand_landmarks):
    landmarks = {i: (lm.x, lm.y) for i, lm in enumerate(hand_landmarks.landmark)}

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]

    # Calculate distances
    thumb_index_dist = euclidean_distance(thumb_tip, index_tip)
    index_middle_dist = euclidean_distance(index_tip, middle_tip)

    # Gesture rules
    if thumb_tip[1] < index_tip[1] < middle_tip[1] and thumb_index_dist > 0.2:
        return 0  # Thumbs Up
    elif thumb_tip[1] > index_tip[1] > middle_tip[1] and thumb_index_dist > 0.2:
        return 1  # peace
    elif index_tip[1] < middle_tip[1] < ring_tip[1] and index_middle_dist > 0.15:
        return 2  # Thumbs down
    elif index_tip[1] < wrist[1] and middle_tip[1] > wrist[1] and thumb_tip[1] > wrist[1]:
        return 3  # Pointing
    elif all(finger[1] > wrist[1] for finger in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]):
        return 4  # Flat Hand
    elif thumb_tip[1] < wrist[1] and middle_tip[1] < wrist[1]:
        return 5  # Thanks (palms together)
    elif all(finger[0] < wrist[0] for finger in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]):
        return 6  # Sorry (hand over chest)
    else:
        return -1  # Unknown Gesture

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for selfie view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Recognize gesture
            gesture_id = recognize_gesture(hand_landmarks)
            gesture_text = gesture_map.get(gesture_id, "Unknown Gesture")

            # Display recognized gesture on frame
            cv2.putText(frame, gesture_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display landmark index for debugging
            for idx, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Show frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
