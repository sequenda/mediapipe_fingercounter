import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# hand object
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# image
image_path = './assets/hands1.jpg'
image = cv2.imread(image_path)

if image is not None:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    while True:
        cv2.imshow('Hand Landmarks', image) 

        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            cv2.destroyAllWindows()
            print("Terminated.")
            break

    cv2.destroyAllWindows()


else:
    print("Image not found.")


