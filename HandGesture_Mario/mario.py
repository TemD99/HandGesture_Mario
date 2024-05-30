import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key
import time

# Initialize MediaPipe Hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize pynput keyboard controller
keyboard = Controller()

# Variables to keep track of the previous gesture and key press status
prev_gesture = None
key_pressed = None
gesture_cooldown = 0.1  # 100 milliseconds between actions
last_action_time = 0

# Function to recognize gestures
def recognize_gesture(landmarks):
    if landmarks:
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        thumb_cmc = landmarks[2]
        index_pip = landmarks[6]
        middle_pip = landmarks[10]
        ring_pip = landmarks[14]
        pinky_pip = landmarks[18]

        # Check for thumbs up (thumb up, other fingers down)
        if (thumb_tip.y < thumb_ip.y and
            index_tip.y > index_pip.y and 
            middle_tip.y > middle_pip.y and 
            ring_tip.y > ring_pip.y and 
            pinky_tip.y > pinky_pip.y):
            print("thumbs_up")
            return "thumbs_up"

        # Check for thumbs down (thumb down, other fingers down)
        if (thumb_tip.y > thumb_ip.y and
            index_tip.y > index_pip.y and 
            middle_tip.y > middle_pip.y and 
            ring_tip.y > ring_pip.y and 
            pinky_tip.y > pinky_pip.y):
            print("thumbs_down")
            return "thumbs_down"

        # Check for fist (all fingers closed)
        if (thumb_tip.y < thumb_cmc.y and
            index_tip.y < index_pip.y and
            middle_tip.y < middle_pip.y and
            ring_tip.y < ring_pip.y and
            pinky_tip.y < pinky_pip.y):
            print("fist")
            return "fist"

        # Check for open hand (all fingers extended)
        if (thumb_tip.y > thumb_ip.y and
            index_tip.y < index_pip.y and 
            middle_tip.y < middle_pip.y and 
            ring_tip.y < ring_pip.y and 
            pinky_tip.y < pinky_pip.y):
            print("open_hand")
            return "open_hand"

    return None

# Function to perform actions based on gestures
def perform_action(gesture):
    global key_pressed, last_action_time
    current_time = time.time()

    if gesture == "thumbs_up":
        if key_pressed != Key.right:
            if key_pressed:
                keyboard.release(key_pressed)
            keyboard.press(Key.right)
            key_pressed = Key.right
    elif gesture == "thumbs_down":
        if key_pressed != Key.left:
            if key_pressed:
                keyboard.release(key_pressed)
            keyboard.press(Key.left)
            key_pressed = Key.left
    elif gesture == "open_hand":
        if key_pressed != 'x':
            if key_pressed:
                keyboard.release(key_pressed)
            keyboard.press('x')
            key_pressed = 'x'
    else:
        if key_pressed:
            keyboard.release(key_pressed)
            key_pressed = None

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gesture = None
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = recognize_gesture(hand_landmarks.landmark)
    
    perform_action(gesture)
    
    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
