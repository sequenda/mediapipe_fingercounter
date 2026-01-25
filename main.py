import mediapipe as mp
import cv2

# tools
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# hand object
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.45)

# video + resolution
videoWidth = 640
videoHeight = 480

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, videoWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, videoHeight)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Landmarks', frame)

    # separated due to niri wm issues
    key = cv2.waitKey(10) & 0xFF

    if key == ord('q'):
        print("Terminated.")
        break
    

else:
    print("Video not found.")


