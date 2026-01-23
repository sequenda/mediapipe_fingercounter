import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# hand object
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.45)

# video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Landmarks', frame)

    key = cv2.waitKey(10) & 0xFF

    if key == ord('q'):
        print("Terminated.")
        break
    

else:
    print("Image not found.")


