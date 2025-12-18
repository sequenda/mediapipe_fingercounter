import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break




cap.release()
cv2.destroyAllWindows()
