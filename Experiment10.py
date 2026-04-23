import cv2 #capture video
import mediapipe as mp #hand tracking
import pyautogui #cursor
import time
import math
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW) #capture camera
my_hand_detector = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75) #detect hand
draw_point = mp.solutions.drawing_utils #draw points
screen_width, screen_height = pyautogui.size() #screen resolution to map with hand coordinate
index_y = 0
smoothed_x, smoothed_y = 0, 0 #better movement of cursor
alpha = 0.3
thumb_is_clicking = False  # Click state
last_click_time = 0  
click_interval = 2.0

while True:
    _, frame = capture.read() #capture each frame from webcam
    frame = cv2.flip(frame, 1) #flip horizontally , mirror effect ke liye
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = my_hand_detector.process(rgb_frame) #process rgb frame to detect hand landmarks
    hands = output.multi_hand_landmarks
    
    if hands:
        for hand in hands:
            draw_point.draw_landmarks(frame, hand) #draw handlanmarks on frame
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                if id == 8:  # Index finger tip
                    cv2.circle(img=frame, center=(x, y), radius=20, color=(0, 255, 255))
                    #map index finger coordinates to screen size
                    index_x = int(x / frame_width * screen_width)
                    index_y = int(y / frame_height * screen_height)

                    # Smoothing cursor movement
                    smoothed_x = (1 - alpha) * smoothed_x + alpha * index_x
                    smoothed_y = (1 - alpha) * smoothed_y + alpha * index_y
                    # Move the cursor smoothly
                    pyautogui.moveTo(smoothed_x, smoothed_y)

                if id == 4:  # Thumb tip
                    cv2.circle(img=frame, center=(x, y), radius=20, color=(0, 255, 255))
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y

                    if abs(index_y - thumb_y) < 30 and not thumb_is_clicking:  # thumb and index within 30 pixels
                        # Check for double-click
                        current_time = time.time()
                        if current_time - last_click_time < click_interval:
                            pyautogui.doubleClick()  # Perform double-click
                        else:
                            pyautogui.click()  # Perform single click
                        last_click_time = current_time  # Update the last click time
                        thumb_is_clicking = True
                        cv2.circle(img=frame, center=(x, y), radius=20, color=(0, 0, 255), thickness=-1)  # Red for click
                    elif abs(index_y - thumb_y) >= 30:
                        thumb_is_clicking = False  # Reset click state

                if id == 20:  
                    cv2.circle(img=frame, center=(x, y), radius=20, color=(0, 255, 255))
                    pinky_x = screen_width/frame_width*x
                    pinky_y = screen_height/frame_height*y
                    
                    if abs(thumb_y - pinky_y) < 30:
                        pyautogui.rightClick()
                        pyautogui.sleep(1)
                
    cv2.imshow('Virtual Cursor', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

capture.release()
cv2.destroyAllWindows()
